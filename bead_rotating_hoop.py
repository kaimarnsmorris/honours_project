import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BeadOnRotatingHoop:
    """
    Simulates a bead on a vertical hoop rotating around its vertical axis.

    The system has one degree of freedom: θ (angle from vertical)
    The hoop rotates with constant angular velocity Ω around the vertical axis.
    """

    def __init__(self, R=1.0, m=1.0, g=9.81, Omega=5.0):
        """
        Parameters:
        R: radius of the hoop
        m: mass of the bead
        g: gravitational acceleration
        Omega: angular velocity of the hoop rotation
        """
        self.R = R
        self.m = m
        self.g = g
        self.Omega = Omega

    def potential_energy(self, theta):
        """Gravitational potential energy"""
        return -self.m * self.g * self.R * np.cos(theta)

    def effective_potential(self, theta):
        """Effective potential including centrifugal force"""
        V_grav = -self.m * self.g * self.R * np.cos(theta)
        V_centrifugal = -0.5 * self.m * self.R**2 * self.Omega**2 * np.sin(theta)**2
        return V_grav + V_centrifugal

    def hamiltonian(self, theta, p_theta):
        """Total energy (Hamiltonian) of the system"""
        kinetic = p_theta**2 / (2 * self.m * self.R**2)
        potential = self.effective_potential(theta)
        return kinetic + potential

    def equations_of_motion(self, theta, p_theta):
        """
        Hamilton's equations:
        dθ/dt = ∂H/∂p_θ
        dp_θ/dt = -∂H/∂θ
        """
        # dθ/dt = p_θ / (m * R²)
        dtheta_dt = p_theta / (self.m * self.R**2)

        # dp_θ/dt = -∂H/∂θ = -dV_eff/dθ
        # V_eff = -mgR cos(θ) - 0.5 * m * R² * Ω² * sin²(θ)
        # dV_eff/dθ = mgR sin(θ) - m * R² * Ω² * sin(θ) * cos(θ)
        dV_dtheta = (self.m * self.g * self.R * np.sin(theta) -
                     self.m * self.R**2 * self.Omega**2 * np.sin(theta) * np.cos(theta))
        dp_dt = -dV_dtheta

        return dtheta_dt, dp_dt

    def euler_integration(self, theta0, p0, dt, num_steps):
        """Euler integration method"""
        theta = np.zeros(num_steps + 1)
        p = np.zeros(num_steps + 1)
        energy = np.zeros(num_steps + 1)

        theta[0] = theta0
        p[0] = p0
        energy[0] = self.hamiltonian(theta0, p0)

        for i in range(num_steps):
            dtheta_dt, dp_dt = self.equations_of_motion(theta[i], p[i])

            theta[i + 1] = theta[i] + dt * dtheta_dt
            p[i + 1] = p[i] + dt * dp_dt
            energy[i + 1] = self.hamiltonian(theta[i + 1], p[i + 1])

        return theta, p, energy

    def leapfrog_integration(self, theta0, p0, dt, num_steps):
        """Leapfrog (Verlet) integration method - symplectic integrator"""
        theta = np.zeros(num_steps + 1)
        p = np.zeros(num_steps + 1)
        energy = np.zeros(num_steps + 1)

        theta[0] = theta0
        p[0] = p0
        energy[0] = self.hamiltonian(theta0, p0)

        # First half-step for momentum
        _, dp_dt = self.equations_of_motion(theta[0], p[0])
        p_half = p[0] + 0.5 * dt * dp_dt

        for i in range(num_steps):
            # Full step for position using half-step momentum
            dtheta_dt = p_half / (self.m * self.R**2)
            theta[i + 1] = theta[i] + dt * dtheta_dt

            # Calculate force at new position
            _, dp_dt = self.equations_of_motion(theta[i + 1], p_half)

            if i < num_steps - 1:
                # Full step for momentum (except last step)
                p_half = p_half + dt * dp_dt
                p[i + 1] = p_half
            else:
                # Final half-step for momentum
                p[i + 1] = p_half + 0.5 * dt * dp_dt

            energy[i + 1] = self.hamiltonian(theta[i + 1], p[i + 1])

        return theta, p, energy

def compare_integrators():
    """Compare Euler and Leapfrog integrators"""

    # System parameters
    bead = BeadOnRotatingHoop(R=1.0, m=1.0, g=9.81, Omega=5.0)

    # Initial conditions
    theta0 = np.pi/4  # 45 degrees from vertical
    p0 = 0.0          # Initially at rest

    # Integration parameters
    dt = 0.01
    t_final = 10.0
    num_steps = int(t_final / dt)
    time = np.linspace(0, t_final, num_steps + 1)

    # Run simulations
    print(f"Running simulations with dt = {dt}, t_final = {t_final}")
    print(f"Initial conditions: θ₀ = {theta0:.3f} rad, p₀ = {p0:.3f}")
    print(f"System parameters: R = {bead.R}, Ω = {bead.Omega:.2f} rad/s")

    theta_euler, p_euler, energy_euler = bead.euler_integration(theta0, p0, dt, num_steps)
    theta_leapfrog, p_leapfrog, energy_leapfrog = bead.leapfrog_integration(theta0, p0, dt, num_steps)

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # Plot 1: Angle vs Time
    ax1.plot(time, theta_euler, 'r--', label='Euler', alpha=0.7)
    ax1.plot(time, theta_leapfrog, 'b-', label='Leapfrog', alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('θ (rad)')
    ax1.set_title('Angle vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Momentum vs Time
    ax2.plot(time, p_euler, 'r--', label='Euler', alpha=0.7)
    ax2.plot(time, p_leapfrog, 'b-', label='Leapfrog', alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('p_θ (kg⋅m²/s)')
    ax2.set_title('Angular Momentum vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase Space (θ vs p_θ)
    ax3.plot(theta_euler, p_euler, 'r--', label='Euler', alpha=0.7)
    ax3.plot(theta_leapfrog, p_leapfrog, 'b-', label='Leapfrog', alpha=0.8)
    ax3.set_xlabel('θ (rad)')
    ax3.set_ylabel('p_θ (kg⋅m²/s)')
    ax3.set_title('Phase Space Trajectory')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Absolute Energy vs Time
    ax4.plot(time, energy_euler, 'r--', label='Euler', alpha=0.7)
    ax4.plot(time, energy_leapfrog, 'b-', label='Leapfrog', alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Energy (J)')
    ax4.set_title('Total Energy vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Energy Conservation Error
    initial_energy = energy_euler[0]

    # Absolute energy error
    energy_error_euler = np.abs(energy_euler - initial_energy)
    energy_error_leapfrog = np.abs(energy_leapfrog - initial_energy)

    ax5.semilogy(time, energy_error_euler, 'r--', label='Euler', alpha=0.7)
    ax5.semilogy(time, energy_error_leapfrog, 'b-', label='Leapfrog', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Absolute Energy Error (J)')
    ax5.set_title('Energy Conservation Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Relative Energy Error (only if initial energy is not too small)
    if np.abs(initial_energy) > 1e-10:
        relative_error_euler = energy_error_euler / np.abs(initial_energy) * 100
        relative_error_leapfrog = energy_error_leapfrog / np.abs(initial_energy) * 100

        ax6.semilogy(time, relative_error_euler, 'r--', label='Euler', alpha=0.7)
        ax6.semilogy(time, relative_error_leapfrog, 'b-', label='Leapfrog', alpha=0.8)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Relative Energy Error (%)')
        ax6.set_title('Relative Energy Conservation Error')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Initial energy too small\nfor relative error calculation',
                horizontalalignment='center', verticalalignment='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Relative Energy Error (N/A)')

    plt.tight_layout()
    plt.show()

    # Print energy conservation statistics
    print("\nEnergy Conservation Analysis:")
    print(f"Initial Energy: {initial_energy:.6f} J")
    print(f"Final Energy (Euler): {energy_euler[-1]:.6f} J")
    print(f"Final Energy (Leapfrog): {energy_leapfrog[-1]:.6f} J")
    print(f"Euler - Max absolute error: {energy_error_euler.max():.2e} J")
    print(f"Leapfrog - Max absolute error: {energy_error_leapfrog.max():.2e} J")
    if np.abs(initial_energy) > 1e-10:
        print(f"Euler - Max relative error: {(energy_error_euler.max()/np.abs(initial_energy)*100):.2e}%")
        print(f"Leapfrog - Max relative error: {(energy_error_leapfrog.max()/np.abs(initial_energy)*100):.2e}%")

    # Analyze equilibrium points
    print("\nEquilibrium Analysis:")
    theta_eq_stable = 0.0  # Bottom of hoop
    theta_eq_unstable = np.pi  # Top of hoop

    # For rotating hoop, there may be additional equilibrium points
    # where gravitational and centrifugal forces balance
    # mg cos(θ) = mR Ω² sin(θ) cos(θ)
    # This gives: g = R Ω² sin(θ) (for θ ≠ 0, π)

    if bead.g < bead.R * bead.Omega**2:
        theta_eq_new = np.arcsin(bead.g / (bead.R * bead.Omega**2))
        print(f"Additional equilibrium points at θ = ±{theta_eq_new:.3f} rad (±{np.degrees(theta_eq_new):.1f}°)")
    else:
        print("No additional equilibrium points (Ω too small)")

def animate_motion():
    """Create an animation of the bead motion"""

    bead = BeadOnRotatingHoop(R=1.0, m=1.0, g=9.81, Omega=5.0)

    # Initial conditions for interesting motion
    theta0 = np.pi/3
    p0 = 0.5

    dt = 0.02
    t_final = 10.0
    num_steps = int(t_final / dt)

    theta, p, energy = bead.leapfrog_integration(theta0, p0, dt, num_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Setup hoop visualization
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Bead on Rotating Hoop')

    # Draw hoop
    circle = plt.Circle((0, 0), bead.R, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle)

    # Bead position
    bead_dot, = ax1.plot([], [], 'ro', markersize=8)

    # Energy plot
    ax2.set_xlim(0, t_final)
    ax2.set_ylim(min(energy) * 1.1, max(energy) * 1.1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Energy vs Time')
    energy_line, = ax2.plot([], [], 'b-')

    def animate(frame):
        if frame < len(theta):
            # Update bead position
            x = bead.R * np.sin(theta[frame])
            y = -bead.R * np.cos(theta[frame])
            bead_dot.set_data([x], [y])

            # Update energy plot
            time_current = np.linspace(0, frame * dt, frame + 1)
            energy_line.set_data(time_current, energy[:frame + 1])

        return bead_dot, energy_line

    anim = FuncAnimation(fig, animate, frames=len(theta), interval=50, blit=True, repeat=True)
    plt.show()

    return anim

if __name__ == "__main__":
    print("Bead on Rotating Hoop Simulation")
    print("=" * 40)

    # Run comparison
    compare_integrators()

    # Create animation
    print("\nCreating animation...")
    anim = animate_motion()