import numpy as np

# constants
epsilon_0: float = 8.854e-12  # electric permittivity of free space
e: float = 1.6022e-19  # electron charge
m_0: float = 9.109e-31  # electron mass


def convert_f_to_omega(f):
    return 2*np.pi*f


def plasma_frequency_squared(N, epsilon_0=epsilon_0, e=e, m_0=m_0):
    """
    The plasma frequency is the square root of (N*e**2)/(epsilon_0*m_0), where N is the
    number of oscillators, e the electric charge, epsilon_0 the vacuum permittivity,
    and m_0 the electron mass.
    Return the square of the plasma frequency.
    """
    return (N * e**2) / (epsilon_0 * m_0)


# constants for Lorentzian approximation
def epsilon_st(omega_0, N, chi, epsilon_0=epsilon_0, e=e, m_0=m_0):
    return 1 + chi + ((N * e**2) / (epsilon_0 * m_0)) / omega_0**2


def epsilon_inf(chi):
    return 1 + chi


# relative dielectric constant
def epsilon_1(
    omega, omega_0, gamma, N, chi, epsilon_0=epsilon_0, e=e, m_0=m_0, approx=False
):  # real part
    if not approx:
        return (
            1
            + chi
            + (
                plasma_frequency_squared(N=N, epsilon_0=epsilon_0, e=e, m_0=m_0)
                * (omega_0**2 - omega**2)
                / ((omega_0**2 - omega**2) ** 2 + (gamma * omega) ** 2)
            )
        )
    else:
        domega = omega - omega_0
        return epsilon_inf(chi) - (
            (
                epsilon_st(
                    omega_0=omega_0, N=N, chi=chi, epsilon_0=epsilon_0, e=e, m_0=m_0
                )
                - epsilon_inf(chi)
            )
            * (2 * omega_0 * domega)
            / (4 * domega**2 + gamma**2)
        )


def epsilon_2(
    omega, omega_0, gamma, N, chi, epsilon_0=epsilon_0, e=e, m_0=m_0, approx=False
):  # imaginary part
    if not approx:
        return plasma_frequency_squared(N=N, epsilon_0=epsilon_0, e=e, m_0=m_0) * (
            gamma * omega / ((omega_0**2 - omega**2) ** 2 + (gamma * omega) ** 2)
        )
    else:
        domega = omega - omega_0
        return (
            (
                epsilon_st(
                    omega_0=omega_0, N=N, chi=chi, epsilon_0=epsilon_0, e=e, m_0=m_0
                )
                - epsilon_inf(chi)
            )
            * gamma
            * omega_0
            / (4 * domega**2 + gamma**2)
        )


# refractive index
def _n(epsilon1, epsilon2):
    return 2 ** (-1 / 2) * np.sqrt(epsilon1 + np.sqrt(epsilon1**2 + epsilon2**2))


def _k(epsilon1, epsilon2):
    return 2 ** (-1 / 2) * np.sqrt(-epsilon1 + np.sqrt(epsilon1**2 + epsilon2**2))


def n(
    omega, omega_0, gamma, N, chi, epsilon_0=epsilon_0, e=e, m_0=m_0, approx=False
):  # real part
    epsilon1 = epsilon_1(
        omega=omega,
        omega_0=omega_0,
        gamma=gamma,
        N=N,
        chi=chi,
        epsilon_0=epsilon_0,
        e=e,
        m_0=m_0,
        approx=approx,
    )
    epsilon2 = epsilon_2(
        omega=omega,
        omega_0=omega_0,
        gamma=gamma,
        N=N,
        chi=chi,
        epsilon_0=epsilon_0,
        e=e,
        m_0=m_0,
        approx=approx,
    )
    return _n(epsilon1, epsilon2)


def k(
    omega, omega_0, gamma, N, chi, epsilon_0=epsilon_0, e=e, m_0=m_0, approx=False
):  # imaginary part
    epsilon1 = epsilon_1(
        omega=omega,
        omega_0=omega_0,
        gamma=gamma,
        N=N,
        chi=chi,
        epsilon_0=epsilon_0,
        e=e,
        m_0=m_0,
        approx=approx,
    )
    epsilon2 = epsilon_2(
        omega=omega,
        omega_0=omega_0,
        gamma=gamma,
        N=N,
        chi=chi,
        epsilon_0=epsilon_0,
        e=e,
        m_0=m_0,
        approx=approx,
    )
    return _k(epsilon1, epsilon2)
