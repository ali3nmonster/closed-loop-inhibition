"""Independent analytical references for the plant/timing validation experiment.

Frequency responses and delayed-PD references describe linear, unsaturated,
fixed-cadence control about a zero reference. They are not transformer models.
"""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov

from .controllers import sampled_closed_loop_matrix
from .plants import Oscillator, _finite_scalar


def _positive(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _delay(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("delay_steps must be a nonnegative integer")
    return int(value)


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return values


def plant_timescales(plant: Oscillator) -> dict:
    """Physical timescales; absent oscillations or decay are explicitly ``None``.

    ``tau`` is inverse natural angular frequency, not the oscillation period.
    The envelope time applies only to underdamped motion with positive damping.
    """
    underdamped = plant.zeta < 1
    omega_d = np.sqrt(1 - plant.zeta**2) / plant.tau if underdamped else None
    return {
        "tau": plant.tau,
        "zeta": plant.zeta,
        "natural_angular_frequency": 1 / plant.tau,
        "natural_period": 2 * np.pi * plant.tau,
        "damped_angular_frequency": float(omega_d) if omega_d is not None else None,
        "damped_period": float(2 * np.pi / omega_d) if omega_d is not None else None,
        "envelope_decay_time": plant.tau / plant.zeta if 0 < plant.zeta < 1 else None,
    }


def free_ringdown(
    plant: Oscillator, times: ArrayLike, initial_state: ArrayLike
) -> NDArray[np.float64]:
    """Exact unforced underdamped state via scalar sine/exponential formulas.

    This reference intentionally does not use the matrix exponential employed
    by the simulator. The returned shape is ``(len(times), 2)``.
    """
    times = _vector(times, "times")
    state = _vector(initial_state, "initial_state")
    if state.shape != (2,):
        raise ValueError("initial_state must have shape (2,)")
    if np.any(times < 0):
        raise ValueError("times must be nonnegative")
    if plant.zeta >= 1:
        raise ValueError("free_ringdown requires an underdamped plant (zeta < 1)")
    decay = plant.zeta / plant.tau
    omega = np.sqrt(1 - plant.zeta**2) / plant.tau
    a, b = state[0], (state[1] + decay * state[0]) / omega
    wave = a * np.cos(omega * times) + b * np.sin(omega * times)
    derivative = omega * (-a * np.sin(omega * times) + b * np.cos(omega * times))
    envelope = np.exp(-decay * times)
    return np.column_stack((envelope * wave, envelope * (derivative - decay * wave)))


def sampled_mode_summary(
    plant: Oscillator, kp: float, kd: float, period: float, delay_steps: int = 0
) -> dict:
    """Poles of the exact sampled map and its dominant amplitude growth rate.

    A positive growth rate denotes instability. ``stable`` requires rho < 1;
    ``classification`` uses a 1e-10 boundary tolerance for roundoff. Exact
    deadbeat rho=0 has growth rate -infinity, serialized as None with an
    explicit ``deadbeat`` flag. Pole angles are principal, sampling-aliased
    mode frequencies and must not be interpreted as unique continuous poles.
    """
    period = _positive(period, "period")
    transition = sampled_closed_loop_matrix(plant, kp, kd, period, delay_steps)
    poles = np.linalg.eigvals(transition)
    rho = float(np.max(np.abs(poles)))
    growth = float(np.log(rho) / period) if rho > 0 else None
    classification = "marginal" if abs(rho - 1) <= 1e-10 else ("stable" if rho < 1 else "unstable")
    return {
        "poles": [{"real": float(p.real), "imag": float(p.imag)} for p in poles],
        "spectral_radius": rho,
        "growth_rate": growth,
        "stable": rho < 1,
        "classification": classification,
        "deadbeat": rho == 0,
        "envelope_time": float(1 / abs(growth)) if growth is not None and growth != 0 else None,
    }


def continuous_force_response(
    plant: Oscillator, angular_frequencies: ArrayLike
) -> NDArray[np.complex128]:
    """Position / physical-force transfer at continuous angular frequencies.

    The phasor convention is exp(+i*omega*t). This is the response to a smooth
    sinusoidal force, not to its sampled-and-held approximation.
    """
    omega = _vector(angular_frequencies, "angular_frequencies")
    if np.any(omega < 0):
        raise ValueError("angular_frequencies must be nonnegative")
    denominator = 1 - (plant.tau * omega) ** 2 + 2j * plant.zeta * plant.tau * omega
    if np.any(denominator == 0):
        raise ValueError("undamped resonant response is singular")
    return np.asarray(1 / denominator, dtype=complex)


def sampled_frequency_response(
    plant: Oscillator,
    kp: float,
    kd: float,
    period: float,
    delay_steps: int,
    angular_frequencies: ArrayLike,
    channel: str = "force",
    force_dt: float | None = None,
) -> NDArray[np.complex128]:
    """Exact sampled-position phasor for held force or capture-time sensor noise.

    ``force`` means d_k is held on [k*h, (k+1)*h), unless ``force_dt`` is
    supplied: then a sinusoid is sampled and held on that finer grid, which
    must divide h exactly. ``position_noise`` means
    n_k is added only to q in the observation captured at k*h. The delayed
    command is u_k=-kp*(q_(k-m)+n_(k-m))-kd*tau*v_(k-m). The solution is
    [zI-Ad+Bd*K*z**(-m)] x = Bd*d - Bd*kp*z**(-m)*n.

    Frequencies must lie in [0, pi/h]. For an unstable map this is only a
    formal particular solution; there is no attracting steady-state response.
    """
    period = _positive(period, "period")
    delay_steps = _delay(delay_steps)
    kp, kd = _finite_scalar(kp, "kp"), _finite_scalar(kd, "kd")
    omega = _vector(angular_frequencies, "angular_frequencies")
    if np.any(omega < 0) or np.any(omega * period > np.pi + 1e-12):
        raise ValueError("angular_frequencies must lie between zero and the Nyquist frequency")
    if channel not in {"force", "position_noise"}:
        raise ValueError("channel must be 'force' or 'position_noise'")
    ad, bd = plant.discretize(period)
    force_dt = period if force_dt is None else _positive(force_dt, "force_dt")
    ratio = period / force_dt
    force_stride = int(round(ratio))
    if force_stride < 1 or not np.isclose(ratio, force_stride, rtol=0, atol=1e-10):
        raise ValueError("period must be an integer multiple of force_dt")
    ad_fine, bd_fine = plant.discretize(force_dt)
    feedback = np.outer(bd, [kp, kd * plant.tau])
    result = np.empty(len(omega), dtype=complex)
    for index, frequency in enumerate(omega):
        z = np.exp(1j * frequency * period)
        matrix = z * np.eye(2) - ad + feedback * z ** (-delay_steps)
        if channel == "force":
            source = np.zeros(2, dtype=complex)
            for fine_step in range(force_stride):
                source = ad_fine @ source + bd_fine * np.exp(1j * frequency * fine_step * force_dt)
        else:
            source = -kp * bd * z ** (-delay_steps)
        try:
            result[index] = np.linalg.solve(matrix, source)[0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("sampled frequency response is singular") from exc
    return result


def ou_open_loop_covariance(
    plant: Oscillator, fine_dt: float, correlation_time: float, std: float
) -> NDArray[np.float64]:
    """Stationary covariance of [position, velocity, held OU force].

    Force d_k has covariance std**2 * exp(-abs(k-j)*dt/correlation_time)
    and acts on the plant during [k*dt,(k+1)*dt). Its AR(1) innovation is
    added at the interval endpoint. This is exact for the held-force model;
    it does not substitute a smooth continuous OU force between samples.
    """
    fine_dt = _positive(fine_dt, "fine_dt")
    correlation_time = _positive(correlation_time, "correlation_time")
    std = _finite_scalar(std, "std")
    if std < 0:
        raise ValueError("std must be nonnegative")
    if plant.zeta <= 0:
        raise ValueError("stationary covariance requires positive plant damping")
    ad, bd = plant.discretize(fine_dt)
    persistence = np.exp(-fine_dt / correlation_time)
    transition = np.zeros((3, 3))
    transition[:2, :2] = ad
    transition[:2, 2] = bd
    transition[2, 2] = persistence
    innovation = np.zeros((3, 3))
    innovation[2, 2] = std**2 * -np.expm1(-2 * fine_dt / correlation_time)
    covariance = solve_discrete_lyapunov(transition, innovation)
    return (covariance + covariance.T) / 2


def delayed_pd_reference(
    plant: Oscillator,
    initial_state: ArrayLike,
    force_values: ArrayLike,
    *,
    fine_dt: float,
    period: float,
    delay_steps: int = 0,
    kp: float = 2.0,
    kd: float = 1.0,
    position_noise: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Independent grid recurrence with exact held-input plant transitions.

    ``force_values[j]`` is held during fine interval j; length N defines the
    duration N*fine_dt. ``period/fine_dt`` must be an integer, and duration an
    integer number of controller periods. Position noise has one value at
    every capture including the final endpoint. It never directly moves the
    physical state. Captures occur at 0,h,...; commands enter a FIFO of length
    ``delay_steps`` with zero commands before the first computed application.

    No actuator saturation or divergence guard is applied. Finite unstable
    trajectories are retained, and floating-point overflow raises explicitly.
    States have N+1 rows and actions have N rows (one per held interval).
    """
    fine_dt, period = _positive(fine_dt, "fine_dt"), _positive(period, "period")
    delay_steps = _delay(delay_steps)
    kp, kd = _finite_scalar(kp, "kp"), _finite_scalar(kd, "kd")
    ratio = period / fine_dt
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=0, atol=1e-10):
        raise ValueError("period must be an integer multiple of fine_dt")
    force = _vector(force_values, "force_values")
    if len(force) == 0 or len(force) % stride:
        raise ValueError("force_values must cover a positive integer number of periods")
    initial = _vector(initial_state, "initial_state")
    if initial.shape != (2,):
        raise ValueError("initial_state must have shape (2,)")
    captures = len(force) // stride + 1
    noise = np.zeros(captures) if position_noise is None else _vector(position_noise, "position_noise")
    if noise.shape != (captures,):
        raise ValueError("position_noise must have one value per capture including final endpoint")
    states = np.empty((len(force) + 1, 2))
    commands = np.empty(captures)
    actions = np.empty(len(force))
    states[0] = initial
    ad, bd = plant.discretize(fine_dt)
    held = 0.0
    for step in range(len(force) + 1):
        if step % stride == 0:
            capture = step // stride
            commands[capture] = -kp * (states[step, 0] + noise[capture]) - kd * plant.tau * states[step, 1]
            held = commands[capture - delay_steps] if capture >= delay_steps else 0.0
        if step == len(force):
            break
        actions[step] = held
        with np.errstate(over="raise", invalid="raise"):
            try:
                states[step + 1] = ad @ states[step] + bd * (held + force[step])
            except FloatingPointError as exc:
                raise FloatingPointError(f"reference trajectory became nonfinite at fine step {step + 1}") from exc
    if not np.all(np.isfinite(commands)):
        raise FloatingPointError("reference commands became nonfinite")
    return {
        "times": np.arange(len(force) + 1) * fine_dt,
        "states": states,
        "actions": actions,
        "capture_times": np.arange(captures) * period,
        "commands": commands,
    }
