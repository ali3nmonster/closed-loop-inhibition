"""Local mechanics of the complete sampled controller--plant feedback loop.

The Poincare section is a mature decision instant, after capture and after any
old completion has applied, but before computing the new command. The state
contains the observation history, held action, and every pending command. For
positive integer-period latency, capture precedes the same-time completion;
for zero latency the new command applies after dispatch. These cases must not
be merged. This is a physical feedback Jacobian, not a transformer layer map.

Frequency responses refer to a force held over each decision period and to
position sampled on the section. They are not responses to a continuous sine
wave. Transient matrix norms use the explicitly documented normalized state
coordinates and are coordinate dependent. Equilibrium results are local and
do not establish stability along noisy trajectories or for finite excursions.
"""

from copy import deepcopy
import math

import numpy as np
from scipy.optimize import brentq
import torch

from .collective_suppression import scale_branches
from .delay_sweep import FixedDelayPolicy, timing_audit
from .imitation import shared_snapshot
from .plants import Oscillator
from .signals import PiecewiseConstant
from .timescale_maps import make_timing
from .timing import simulate


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


class EventCycleMap:
    """Differentiable mature-cycle map with exact held-input plant propagation.

    State ordering is oldest-to-newest history triples
    ``(q/state_scale, tau*v/state_scale, captured_action/output_scale)``,
    followed by held action/output_scale and pending commands/output_scale in
    increasing application-time order. The latest history already contains the
    current plant state. Reference and reference velocity are fixed at zero;
    observation cadence equals decision cadence, with no sensor/actuator delay.

    All learned float32 weights are copied, then represented in float64. The
    mathematical smooth extension is analyzed; parity with the original float32
    event simulator is checked separately, including its feature rounding.
    """

    def __init__(self, model, base, tau, delay, settings=None, delay_cue=.05):
        self.base = deepcopy(base)
        self.period = _positive(base["period"], "period")
        self.tau = _positive(tau, "tau")
        self.delay = _positive(delay, "delay", zero=True)
        self.delay_cue = _positive(delay_cue, "delay_cue", zero=True)
        self.state_scale = _positive(base["state_scale"], "state_scale")
        self.output_scale = _positive(base["output_scale"], "output_scale")
        self.action_limit = _positive(base["action_limit"], "action_limit")
        self.plant = Oscillator(self.tau, base["zeta"])
        h_ticks = round(self.period * 1e9)
        l_ticks = round(self.delay * 1e9)
        if h_ticks < 1 or abs(h_ticks / 1e9 - self.period) > 1e-12:
            raise ValueError("period must be an exact nanosecond clock value")
        if abs(l_ticks / 1e9 - self.delay) > 1e-12:
            raise ValueError("delay must be an exact nanosecond clock value")
        self.whole_periods, r_ticks = divmod(l_ticks, h_ticks)
        self.remainder = r_ticks / 1e9
        self.pending_count = self.whole_periods - (r_ticks == 0 and l_ticks > 0)
        self.current_action_age = (0. if l_ticks and not r_ticks else
                                   self.period - self.remainder)
        count = base["max_tokens"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("max_tokens must be a positive integer")
        history_ticks = round(base["history_seconds"] * 1e9)
        self.history_count = min(count, history_ticks // h_ticks + 1)
        self.dimension = 3 * self.history_count + 1 + self.pending_count
        self.position_index = 3 * (self.history_count - 1)
        self.settings = deepcopy(settings or {})
        self.offset = float(self.settings.get("offset", 0.))
        self.center = float(self.settings.get("center", 0.))
        self.gain = _positive(self.settings.get("gain", 1.), "gain", zero=True)
        if not np.isfinite(self.offset) or not np.isfinite(self.center):
            raise ValueError("offset and center must be finite")
        scales = self.settings.get("branch_scales", {})
        self.float_model = (scale_branches(model, scales) if scales else deepcopy(model)).eval()
        self.model = deepcopy(self.float_model).double().eval().requires_grad_(True)
        # Keeping requires_grad on parameters avoids the eval-only fused path
        # for which PyTorch does not provide an input derivative in all builds.
        self._transitions = {}
        for dt in (self.period, self.remainder, self.period - self.remainder):
            ad, bd = self.plant.discretize(dt)
            scaling = np.diag([self.state_scale, self.state_scale / self.tau])
            normalized_ad = np.linalg.solve(scaling, ad @ scaling)
            normalized_bd = np.linalg.solve(scaling, bd)
            self._transitions[dt] = (torch.tensor(normalized_ad, dtype=torch.float64),
                                     torch.tensor(normalized_bd, dtype=torch.float64))

    def _state(self, state):
        state = torch.as_tensor(state, dtype=torch.float64)
        if state.ndim != 1 or state.numel() != self.dimension:
            raise ValueError(f"state must be a vector of length {self.dimension}")
        return state

    def tokens(self, state):
        state = self._state(state)
        history = state[:3 * self.history_count].reshape(self.history_count, 3)
        held = state[3 * self.history_count]
        zeros = torch.zeros(self.history_count, dtype=state.dtype)
        ages = torch.arange(self.history_count - 1, -1, -1, dtype=state.dtype) * self.period / self.tau
        tokens = torch.stack((history[:, 0], history[:, 1], zeros, zeros,
                              history[:, 2], ages, zeros,
                              zeros + self.delay_cue / self.tau,
                              zeros + held, zeros + self.current_action_age / self.tau), dim=1)
        # The original model uses learned positions for an 11-token padded
        # array. Keep that shape even if a shorter history budget was supplied.
        padding = self.base["max_tokens"] - self.history_count
        if padding:
            tokens = torch.cat((tokens, torch.zeros(padding, 10, dtype=state.dtype)))
        valid = torch.arange(self.base["max_tokens"]) < self.history_count
        return tokens, valid

    def raw_command(self, state):
        tokens, valid = self.tokens(state)
        raw = self.model(tokens[None], valid[None])[0] * self.output_scale
        return self.center + self.gain * (raw - self.center) + self.offset

    def command(self, state):
        return torch.clamp(self.raw_command(state), -self.action_limit, self.action_limit)

    def _propagate(self, plant_state, dt, action, disturbance):
        ad, bd = self._transitions[dt]
        return ad @ plant_state + bd * (action * self.output_scale + disturbance)

    def step(self, state, disturbance=0.):
        """Advance one period; ``disturbance`` is a physical constant force."""
        state = self._state(state)
        return self._step_with_command(state, self.command(state), disturbance)

    def _step_with_command(self, state, physical_command, disturbance):
        history = state[:3 * self.history_count].reshape(self.history_count, 3)
        held = state[3 * self.history_count]
        pending = state[3 * self.history_count + 1:]
        command = physical_command / self.output_scale
        incoming = torch.cat((pending, command[None]))
        qv = history[-1, :2]
        if self.delay == 0.:
            qv = self._propagate(qv, self.period, command, disturbance)
            captured, next_held, next_pending = command, command, pending
        elif self.remainder == 0.:
            qv = self._propagate(qv, self.period, held, disturbance)
            # The observation at the right boundary is captured BEFORE the
            # old inference completes and applies its result at that boundary.
            captured, next_held, next_pending = held, incoming[0], incoming[1:]
        else:
            qv = self._propagate(qv, self.remainder, held, disturbance)
            next_held = incoming[0]
            qv = self._propagate(qv, self.period - self.remainder, next_held, disturbance)
            captured, next_pending = next_held, incoming[1:]
        next_history = torch.cat((history[1:], torch.cat((qv, captured[None]))[None]))
        return torch.cat((next_history.flatten(), next_held[None], next_pending))

    def uniform_state(self, position, disturbance=0.):
        """A candidate stationary state: q=u+d and v=0, with full history."""
        action = float(position) - float(disturbance)
        triple = [position / self.state_scale, 0., action / self.output_scale]
        return np.asarray(triple * self.history_count
                          + [action / self.output_scale] * (1 + self.pending_count))

    def equilibrium(self):
        """Find a zero-force equilibrium on the full actuator-bounded bracket.

        Brent's method returns one root in this bracket; uniqueness is not
        assumed. Every variant has its own operating point, including offsets.
        """
        def residual(position):
            with torch.no_grad():
                return float(self.command(self.uniform_state(position))) - position
        q, report = brentq(residual, -self.action_limit, self.action_limit,
                          xtol=1e-13, rtol=1e-12, full_output=True)
        state = self.uniform_state(q)
        with torch.no_grad():
            error = float(np.max(np.abs(self.step(state).numpy() - state)))
            raw = float(self.raw_command(state))
        distance_to_kink = abs(abs(raw) - self.action_limit)
        return state, {"position": float(q), "velocity": 0., "action": float(q),
                       "converged": bool(report.converged), "state_residual_max": error,
                       "method": "brent_full_actuator_bracket; uniqueness_not_assumed",
                       "raw_command": raw, "clipped": bool(abs(raw) > self.action_limit),
                       "distance_to_clipping_kink": distance_to_kink,
                       "differentiable": bool(distance_to_kink > 1e-8)}

    def linearize(self, state):
        state = self._state(state).detach().requires_grad_(True)
        disturbance = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        a, b = torch.autograd.functional.jacobian(self.step, (state, disturbance))
        k = torch.autograd.functional.jacobian(self.command, state)
        c = np.zeros(self.dimension)
        c[self.position_index] = self.state_scale
        return a.detach().numpy(), b.detach().numpy(), c, k.detach().numpy()

    def from_snapshot(self, snapshot, jobs):
        """Reconstruct the mature section from a real event-simulator snapshot."""
        bounded = shared_snapshot(snapshot, self.base["max_tokens"], self.base["history_seconds"])
        if len(bounded.observations) != self.history_count:
            raise ValueError("snapshot has not accumulated a complete history")
        time = snapshot.time
        expected_times = time - np.arange(self.history_count - 1, -1, -1) * self.period
        observations = bounded.observations
        if not np.allclose([o.capture_time for o in observations], expected_times, atol=1e-10, rtol=0.):
            raise ValueError("snapshot does not lie on the required observation section")
        if any(o.available_time != o.capture_time or o.reference or o.reference_velocity for o in observations):
            raise ValueError("map requires zero sensor delay and zero reference")
        current = bounded.actions[-1]
        if not math.isclose(time - current.time, self.current_action_age, abs_tol=2e-9):
            raise ValueError("action pipeline has not reached its repeating phase")
        future = sorted((j for j in jobs if j["start_time"] < time - 1e-10
                         and j["apply_time"] > time + 1e-10), key=lambda j: j["apply_time"])
        if len(future) != self.pending_count:
            raise ValueError("pending command count differs from the repeating map")
        expected_phase = self.remainder if self.remainder else self.period
        if not np.allclose([j["apply_time"] - time for j in future],
                           expected_phase + np.arange(self.pending_count) * self.period,
                           atol=2e-9, rtol=0.):
            raise ValueError("pending commands do not have the required application phases")
        state = []
        for obs in observations:
            state.extend((obs.position / self.state_scale, self.tau * obs.velocity / self.state_scale,
                          obs.applied_action / self.output_scale))
        state.append(current.value / self.output_scale)
        state.extend(j["action"] / self.output_scale for j in future)
        return np.asarray(state)


def simulator_parity(cycle, *, duration=2., tolerance=2e-6, command_tolerance=1e-6):
    """One-step map parity on every mature snapshot of a float32 real rollout.

    This independently exercises the original encoder, event scheduler, plant
    propagation, fractional and integer application order, and pending queue.
    Commands and state errors include the declared float32-to-float64 change.
    """
    config = deepcopy(cycle.base)
    config["delay"] = cycle.delay
    duration = max(float(duration), cycle.base["history_seconds"] + cycle.delay + 3 * cycle.period)
    n_steps = int(math.ceil(duration / cycle.period))
    duration = n_steps * cycle.period
    disturbance = PiecewiseConstant(0., ((.75, .02), (.85, 0.)))
    if abs(.75 / cycle.period - round(.75 / cycle.period)) > 1e-10 or abs(.1 / cycle.period - round(.1 / cycle.period)) > 1e-10:
        raise ValueError("parity pulse boundaries must coincide with decision sections")
    policy = FixedDelayPolicy(cycle.float_model, cycle.plant, config,
                             offset=cycle.offset, gain=cycle.gain, center=cycle.center,
                             delay_cue=cycle.delay_cue)
    snapshots = []

    def capture(snapshot):
        snapshots.append(snapshot)
        return policy(snapshot)

    run = simulate(cycle.plant, capture, make_timing(config, duration, cycle.period),
                   initial_state=(.005, -.01), disturbance=disturbance)
    audit = timing_audit(run, cycle.period, cycle.delay)
    state_errors, command_errors = [], []
    for index, snapshot in enumerate(snapshots[:-1]):
        if snapshot.time < max(cycle.base["history_seconds"], cycle.delay + cycle.period) - 1e-10:
            continue
        state = cycle.from_snapshot(snapshot, run.jobs)
        following = cycle.from_snapshot(snapshots[index + 1], run.jobs)
        with torch.no_grad():
            predicted = cycle.step(state, disturbance.at(snapshot.time)).numpy()
            command = float(cycle.command(state))
        state_errors.append(float(np.max(np.abs(predicted - following))))
        command_errors.append(abs(command - run.jobs[index]["action"]))
    max_state = max(state_errors, default=None)
    max_command = max(command_errors, default=None)
    passed = bool(state_errors and max_state <= tolerance and max_command <= command_tolerance
                  and run.failure_reason is None)
    return {"passed": passed, "physical_rollouts": 1, "duration": duration,
            "censored": run.failure_reason is not None, "failure_reason": run.failure_reason,
            "mature_steps_compared": len(state_errors), "state_error_max_normalized": max_state,
            "command_error_max_physical": max_command, "state_tolerance_normalized": tolerance,
            "command_tolerance_physical": command_tolerance, "timing": audit,
            "initial_state": [.005, -.01], "pulse": {"onset": .75, "width": .1, "amplitude": .02}}


def _complex(value):
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def linear_analysis(a, b, c, period, frequencies, steps):
    """Spectra and transfer of the complete augmented sampled system."""
    poles = np.linalg.eigvals(a)
    order = np.argsort(-np.abs(poles))
    poles = poles[order]
    dominant = poles[0]
    rho = float(abs(dominant))
    decay = -math.log(rho) / period if rho > 0 else None
    frequency = abs(float(np.angle(dominant))) / (2 * math.pi * period)
    continuous = np.log(complex(dominant)) / period if rho > 0 else None
    damping_ratio = (-float(continuous.real) / abs(continuous)
                     if continuous is not None and abs(continuous) > 0 else None)
    stable = rho < 1.
    transfer = []
    for frequency_value in frequencies:
        frequency_value = _positive(frequency_value, "frequency", zero=True)
        if frequency_value >= .5 / period:
            raise ValueError("frequencies must be below the decision-grid Nyquist frequency")
        z = np.exp(2j * np.pi * frequency_value * period)
        system = z * np.eye(len(a)) - a
        condition = float(np.linalg.cond(system))
        if not np.isfinite(condition) or condition > 1e14:
            transfer.append({"frequency_hz": frequency_value, "available": False,
                             "reason": "near_singular_resolvent",
                             "condition_number": condition if np.isfinite(condition) else None})
            continue
        value = c @ np.linalg.solve(system, b)
        transfer.append({"frequency_hz": frequency_value, "available": True,
                         "real": float(value.real), "imag": float(value.imag),
                         "magnitude": float(abs(value)), "phase_degrees": float(np.angle(value, deg=True)),
                         "condition_number": condition,
                         "steady_state_interpretation_valid": stable})
    power = np.eye(len(a))
    peaks = []
    response = np.zeros(len(a))
    position = [0.]
    for n in range(steps):
        power = a @ power
        peaks.append(float(np.linalg.norm(power, 2)))
        response = a @ response + (b if n == 0 else 0.)
        position.append(float(c @ response))
    return {"spectral_radius": rho, "locally_asymptotically_stable": bool(stable),
            "dominant_decay_rate_per_s": decay, "dominant_frequency_hz": frequency,
            "dominant_damping_ratio": damping_ratio,
            "dominant_pole": _complex(dominant), "poles": [_complex(p) for p in poles],
            "frequency_response": transfer,
            "frequency_convention": "physical force ZOH over h; position sampled at pre-dispatch sections; C(zI-A)^-1B",
            "transient_norm_coordinate_system": "q/state_scale, tau*v/state_scale, actions/output_scale; redundant history included",
            "transient_norm_max": max([1.] + peaks),
            "transient_norm_peak_time": 0. if not peaks or max(peaks) <= 1. else (int(np.argmax(peaks)) + 1) * period,
            "transient_norm": [1.] + peaks,
            "unit_one_period_force_position": position}


def impulse_analysis(cycle, equilibrium, a, b, c, *, amplitudes, width, duration,
                     verification_amplitude=1e-6, settling_fraction=.02):
    """Local and finite-amplitude ring-down from each variant's own equilibrium.

    This uses the full nonlinear cycle map and deterministic zero background.
    No noisy task trials or independent physical simulator runs are implied.
    Positive and negative pulses are both retained. Energy is explicitly a
    decision-sampled squared-position integral, not mechanical energy.
    """
    steps = round(duration / cycle.period)
    width_steps = round(width / cycle.period)
    if steps < 2 or width_steps < 1 or width_steps >= steps:
        raise ValueError("impulse duration and width need positive integral period counts")
    if abs(steps * cycle.period - duration) > 1e-9 or abs(width_steps * cycle.period - width) > 1e-9:
        raise ValueError("impulse duration and width must be integer multiples of period")
    amplitudes = [_positive(v, "impulse_amplitude") for v in amplitudes]
    verification_amplitude = _positive(verification_amplitude, "verification_amplitude")
    if not 0 < settling_fraction < 1:
        raise ValueError("settling_fraction must lie between zero and one")
    if len(set(amplitudes)) != len(amplitudes):
        raise ValueError("impulse amplitudes must be unique")
    ordered = list(dict.fromkeys([verification_amplitude] + amplitudes))
    records = []
    for magnitude in ordered:
        for sign in (-1, 1):
            amplitude = sign * magnitude
            nonlinear = torch.tensor(equilibrium, dtype=torch.float64)
            linear = np.zeros(cycle.dimension)
            nonlinear_q, linear_q, action_change = [0.], [0.], []
            clipped_count = 0
            censored = False
            for n in range(steps):
                force = amplitude if n < width_steps else 0.
                with torch.no_grad():
                    raw = cycle.raw_command(nonlinear)
                    clipped_count += abs(float(raw)) >= cycle.action_limit
                    command = torch.clamp(raw, -cycle.action_limit, cycle.action_limit)
                    action_change.append(float(command) - equilibrium[cycle.position_index] * cycle.state_scale)
                    nonlinear = cycle._step_with_command(nonlinear, command, force)
                linear = a @ linear + b * force
                nonlinear_q.append(float((nonlinear[cycle.position_index] - equilibrium[cycle.position_index]) * cycle.state_scale))
                linear_q.append(float(c @ linear))
                if not np.isfinite(nonlinear.numpy()).all() or np.max(np.abs(nonlinear.numpy())) > 1e6:
                    censored = True
                    break
            nq, lq = np.asarray(nonlinear_q), np.asarray(linear_q)
            error = (float(np.linalg.norm(nq - lq) / np.linalg.norm(lq))
                     if not censored and np.linalg.norm(lq) > 1e-20 else None)
            short_steps = min(len(lq), round(.5 / cycle.period) + 1)
            short_error = (float(np.linalg.norm(nq[:short_steps] - lq[:short_steps]) / np.linalg.norm(lq[:short_steps]))
                           if np.linalg.norm(lq[:short_steps]) > 1e-20 else None)
            if censored:
                energy = peak = settling = crossings = None
            else:
                energy = float(np.trapezoid((nq / amplitude) ** 2, dx=cycle.period))
                peak = float(np.max(np.abs(nq)) / abs(amplitude))
                tail = nq[width_steps:]
                meaningful = tail[np.abs(tail) > max(np.max(np.abs(nq)) * 1e-5, 1e-14)]
                crossings = int(np.sum(meaningful[1:] * meaningful[:-1] < 0))
                outside = np.flatnonzero(np.abs(nq[width_steps:]) > np.max(np.abs(nq)) * settling_fraction)
                # Null means recovery to the 2% peak band was not observed.
                settling = (float((outside[-1] + 1) * cycle.period)
                            if len(outside) and outside[-1] < len(nq) - width_steps - 1
                            else 0. if not len(outside) else None)
            records.append({"amplitude": amplitude, "verification_probe": magnitude == verification_amplitude,
                            "censored": censored, "observed_duration": (len(nonlinear_q) - 1) * cycle.period,
                            "position_response": nonlinear_q,
                            "linear_position_response": linear_q, "command_change": action_change,
                            "relative_linear_prediction_error": error,
                            "first_half_second_relative_linear_prediction_error": short_error,
                            "sampled_recovery_integral_normalized": energy,
                            "peak_position_over_amplitude": peak,
                            "post_pulse_sign_crossings": crossings,
                            "settling_time_after_pulse_2pct_peak": settling,
                            "clipped_decision_count": int(clipped_count)})
    verification = [r for r in records if r["verification_probe"]]
    return {"force_width": width, "duration": duration, "period": cycle.period,
            "times": (np.arange(steps + 1) * cycle.period).tolist(),
            "records": records,
            "verification_amplitude": verification_amplitude,
            "verification_duration": .5,
            "verification_relative_error_max": max(r["first_half_second_relative_linear_prediction_error"] for r in verification)
            if all(r["first_half_second_relative_linear_prediction_error"] is not None for r in verification) else None,
            "settling_fraction": settling_fraction,
            "metric_convention": "trapezoidal q-squared on decision sections / physical pulse amplitude squared; zero-background own equilibrium"}


def analyze_model(base, protocol, tau, noise_tau, seed, model, prepared, delay):
    """Analyze every available frozen or freshly calibrated variant in a cell."""
    spec = protocol.get("mechanism", {})
    duration = float(spec.get("duration", 12.))
    frequencies = spec.get("frequencies", [.25, .5, 1., 2., 4., 8.])
    configured_amplitudes = spec.get("pulse_amplitudes", [-.02, -.0001, .0001, .02])
    if any(not np.isfinite(value) or value == 0 for value in configured_amplitudes):
        raise ValueError("pulse amplitudes must be finite and nonzero")
    amplitudes = sorted(set(abs(float(value)) for value in configured_amplitudes))
    if set(configured_amplitudes) != {sign * value for value in amplitudes for sign in (-1, 1)}:
        raise ValueError("mechanism pulse amplitudes must be paired positive and negative")
    width = float(spec.get("pulse_width", .1))
    verification_amplitude = float(spec.get("verification_amplitude", 1e-6))
    tolerance = float(spec.get("linearization_relative_tolerance", .01))
    results = {}
    physical_rollouts = 0
    for name, settings in prepared["variants"].items():
        cycle = EventCycleMap(model, base, tau, delay, settings, protocol.get("delay_cue", base["delay"]))
        parity = simulator_parity(cycle, duration=spec.get("parity_duration", 2.),
                                  tolerance=spec.get("parity_state_tolerance", 2e-6),
                                  command_tolerance=spec.get("parity_command_tolerance", 1e-6))
        physical_rollouts += parity["physical_rollouts"]
        if not parity["passed"]:
            raise AssertionError(f"Full-loop map does not match event simulator for {name}: {parity}")
        try:
            equilibrium, operating_point = cycle.equilibrium()
            a, b, c, k = cycle.linearize(equilibrium)
            linear = linear_analysis(a, b, c, cycle.period, frequencies, round(duration / cycle.period))
            impulses = impulse_analysis(cycle, equilibrium, a, b, c, amplitudes=amplitudes,
                                        width=width, duration=duration,
                                        verification_amplitude=verification_amplitude,
                                        settling_fraction=spec.get("settling_fraction", .02))
            relative_error = impulses["verification_relative_error_max"]
            issues = []
            if not operating_point["converged"] or operating_point["state_residual_max"] > spec.get("equilibrium_tolerance", 1e-9):
                issues.append("equilibrium_not_resolved")
            if not operating_point["differentiable"]:
                issues.append("equilibrium_at_clipping_kink")
            if relative_error is None or relative_error > tolerance:
                issues.append("finite_horizon_tiny_pulse_linear_prediction_failed")
            results[name] = {"status": "complete" if not issues else "numerical_failure",
                             "numerical_issues": issues, "dimension": cycle.dimension,
                             "history_count": cycle.history_count, "pending_command_count": cycle.pending_count,
                             "current_action_age": cycle.current_action_age,
                             "remainder_delay": cycle.remainder, "settings": deepcopy(settings),
                             "equilibrium": operating_point, "equilibrium_state": equilibrium.tolist(),
                             "state_jacobian": a.tolist(), "force_input_jacobian": b.tolist(),
                             "position_readout": c.tolist(), "command_jacobian": k.tolist(),
                             "linear": linear, "impulses": impulses, "simulator_parity": parity}
        except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError) as error:
            results[name] = {"status": "numerical_failure", "numerical_issues": [str(error)],
                             "settings": deepcopy(settings), "simulator_parity": parity}
    return {"tau": float(tau), "noise_tau": float(noise_tau), "seed": int(seed), "delay": float(delay),
            "period": float(base["period"]), "delay_cue": float(protocol.get("delay_cue", base["delay"])),
            "status": "complete" if results and all(r["status"] == "complete" for r in results.values()) else "numerical_failure",
            "physical_rollouts": physical_rollouts, "variants": results,
            "scope": "local stationary complete-loop mechanism and zero-background finite-pulse map; no noisy-trajectory stability claim"}
