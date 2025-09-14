"""
Bead on Spinning Hoop Simulation with Animation
==============================================

This script simulates a bead sliding on a vertically oriented hoop that rotates 
about the vertical axis. It compares Euler and Leapfrog integration methods.

The animation shows:
- Left panel: Leapfrog method (blue bead)  
- Right panel: Euler method (red bead)
- Bottom left: Phase space trajectories
- Bottom right: Energy conservation over time

Key features:
- Real-time visualization of bead motion
- Trailing paths to show recent motion
- Live phase space and energy plots
- Clear demonstration of energy conservation differences

To save the animation as a GIF, uncomment the animation.save() line in main.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def leapfrog_history(q0, p0, dt, num_steps, q_dot, p_dot):
    """
    Leapfrog integration for Hamiltonian system
    q_dot and p_dot are functions that return derivatives given (q, p, t)
    """
    q_history = np.zeros(num_steps + 1)
    p_history = np.zeros(num_steps + 1)
    t_history = np.zeros(num_steps + 1)
    
    q_history[0] = q0
    p_history[0] = p0
    t_history[0] = 0.0
    
    q, p = q0, p0
    
    for i in range(num_steps):
        t = i * dt
        
        # Half step for momentum
        p_half = p + 0.5 * dt * p_dot(q, p, t)
        
        # Full step for position
        q_new = q + dt * q_dot(q, p_half, t + 0.5 * dt)
        
        # Half step for momentum
        p_new = p_half + 0.5 * dt * p_dot(q_new, p_half, t + dt)
        
        q, p = q_new, p_new
        q_history[i + 1] = q
        p_history[i + 1] = p
        t_history[i + 1] = (i + 1) * dt
    
    return t_history, q_history, p_history

def euler_history(q0, p0, dt, num_steps, q_dot, p_dot):
    """
    Euler integration for Hamiltonian system
    q_dot and p_dot are functions that return derivatives given (q, p, t)
    """
    q_history = np.zeros(num_steps + 1)
    p_history = np.zeros(num_steps + 1)
    t_history = np.zeros(num_steps + 1)
    
    q_history[0] = q0
    p_history[0] = p0
    t_history[0] = 0.0
    
    q, p = q0, p0
    
    for i in range(num_steps):
        t = i * dt
        
        # Euler step
        q_new = q + dt * q_dot(q, p, t)
        p_new = p + dt * p_dot(q, p, t)
        
        q, p = q_new, p_new
        q_history[i + 1] = q
        p_history[i + 1] = p
        t_history[i + 1] = (i + 1) * dt
    
    return t_history, q_history, p_history

def bead_on_hoop_dynamics(omega):
    """
    Returns the equations of motion for a bead on a spinning hoop
    
    For a bead on a vertically oriented hoop of radius R, mass m, 
    spinning with angular velocity omega about the vertical axis:
    
    Hamiltonian: H = p²/(2mR²) - mgR*cos(θ) - (1/2)*m*R²*ω²*sin²(θ)
    
    With g = R = m = 1:
    H = p²/2 - cos(θ) - (1/2)*ω²*sin²(θ)
    
    Hamilton's equations:
    dθ/dt = ∂H/∂p = p
    dp/dt = -∂H/∂θ = -sin(θ) - ω²*sin(θ)*cos(θ)
    """
    
    def q_dot(q, p, t):
        # dθ/dt = p (since we set mR² = 1)
        return p
    
    def p_dot(q, p, t):
        # dp/dt = -sin(θ) - ω²*sin(θ)*cos(θ)
        return -np.sin(q) - omega**2 * np.sin(q) * np.cos(q)
    
    return q_dot, p_dot

def hamiltonian(q, p, omega):
    """Calculate the Hamiltonian (total energy) of the system"""
    return 0.5 * p**2 - np.cos(q) - 0.5 * omega**2 * np.sin(q)**2

def find_equilibrium_angles(omega):
    """
    Find equilibrium angles for given omega.
    
    Equilibrium occurs when dp/dt = 0:
    -sin(θ) - ω²*sin(θ)*cos(θ) = 0
    sin(θ)*(1 + ω²*cos(θ)) = 0
    
    Solutions:
    1) sin(θ) = 0  →  θ = 0, π (unstable for ω > 0)
    2) 1 + ω²*cos(θ) = 0  →  cos(θ) = -1/ω² (stable if |1/ω²| ≤ 1, i.e., ω ≥ 1)
    """
    equilibria = []
    
    # θ = 0 is always an equilibrium (unstable for ω > 0)
    equilibria.append((0.0, "unstable"))
    
    # θ = π is always an equilibrium (unstable)
    equilibria.append((np.pi, "unstable"))
    
    # Stable equilibria exist only if ω ≥ 1
    if omega >= 1.0:
        cos_theta_eq = -1.0 / (omega**2)
        if abs(cos_theta_eq) <= 1.0:  # Valid solution
            theta_eq = np.arccos(cos_theta_eq)
            equilibria.append((theta_eq, "stable"))
            equilibria.append((-theta_eq, "stable"))
            # Also the 2π periodic versions
            equilibria.append((2*np.pi - theta_eq, "stable"))
    
    return equilibria

def effective_potential(theta, omega):
    """
    Calculate the effective potential V_eff(θ) = -cos(θ) - (1/2)*ω²*sin²(θ)
    The bead oscillates around minima of this potential.
    """
    return -np.cos(theta) - 0.5 * omega**2 * np.sin(theta)**2

def find_initial_conditions_for_equilibrium(omega, theta_eq_target=None, oscillation_amplitude=0.1):
    """
    Find initial conditions that will keep the bead near an equilibrium position.
    
    Parameters:
    - omega: angular velocity of hoop
    - theta_eq_target: target equilibrium angle (if None, finds stable one)
    - oscillation_amplitude: how far from equilibrium to start (small oscillations)
    """
    
    # Find equilibrium angles
    equilibria = find_equilibrium_angles(omega)
    stable_equilibria = [eq for eq in equilibria if eq[1] == "stable"]
    
    if len(stable_equilibria) == 0:
        print(f"No stable equilibria exist for ω = {omega:.2f}")
        print("Need ω ≥ 1 for stable equilibria away from θ = 0")
        return None
    
    # Choose target equilibrium
    if theta_eq_target is None:
        # Choose the first positive stable equilibrium
        theta_eq = stable_equilibria[0][0]
    else:
        # Find closest equilibrium to target
        theta_eq = min(stable_equilibria, key=lambda x: abs(x[0] - theta_eq_target))[0]
    
    print(f"Target equilibrium: θ = {theta_eq:.3f} rad ({theta_eq*180/np.pi:.1f}°)")
    print(f"This is {theta_eq*180/np.pi:.1f}° measured counterclockwise from the bottom of the hoop")
    if theta_eq > 0:
        print("(Right side of hoop when viewed from front)")
    
    # Initial conditions for small oscillations around equilibrium
    q0 = theta_eq + oscillation_amplitude  # Start slightly displaced
    p0 = 0.0  # Start from rest
    
    # Calculate the energy at this point
    E_target = hamiltonian(q0, p0, omega)
    print(f"Energy level: E = {E_target:.4f}")
    
    return q0, p0, theta_eq

def plot_effective_potential(omega_values, theta_range=np.linspace(-np.pi, 2*np.pi, 1000)):
    """Plot the effective potential for different omega values"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, omega in enumerate(omega_values):
        ax = axes[i]
        
        V_eff = effective_potential(theta_range, omega)
        ax.plot(theta_range, V_eff, 'b-', linewidth=2, label=f'ω = {omega}')
        
        # Mark equilibria
        equilibria = find_equilibrium_angles(omega)
        for theta_eq, stability in equilibria:
            if -np.pi <= theta_eq <= 2*np.pi:  # Only plot in our range
                V_eq = effective_potential(theta_eq, omega)
                color = 'ro' if stability == "stable" else 'rx'
                label = f'{stability} eq.'
                ax.plot(theta_eq, V_eq, color, markersize=8, 
                       label=label if theta_eq == equilibria[0][0] else "")
        
        ax.set_xlabel('θ (rad)')
        ax.set_ylabel('V_eff(θ)')
        ax.set_title(f'Effective Potential (ω = {omega})')
        ax.grid(True)
        ax.legend()
        
        # Add theta labels
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'])
    
    plt.tight_layout()
    plt.suptitle('Effective Potential for Different Angular Velocities', y=1.02)
    return fig

def animate_comparison(omega, dt, num_steps, q0, p0, speed_factor=1):
    """Create an animated comparison of Leapfrog vs Euler methods"""
    
    # Get dynamics functions
    q_dot, p_dot = bead_on_hoop_dynamics(omega)
    
    # Run simulations
    t_leap, q_leap, p_leap = leapfrog_history(q0, p0, dt, num_steps, q_dot, p_dot)
    t_euler, q_euler, p_euler = euler_history(q0, p0, dt, num_steps, q_dot, p_dot)
    
    # Set up the figure and axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hoop parameters (radius = 1)
    R = 1.0
    
    # Set up hoop visualization (left: Leapfrog, right: Euler)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Leapfrog Method', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Euler Method', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Draw the hoops
    hoop1 = patches.Circle((0, 0), R, fill=False, color='black', linewidth=3)
    hoop2 = patches.Circle((0, 0), R, fill=False, color='black', linewidth=3)
    ax1.add_patch(hoop1)
    ax2.add_patch(hoop2)
    
    # Add vertical axis indicators and labels
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0, -1.3, 'θ=0\n(bottom)', ha='center', va='top', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax1.text(0, 1.3, 'θ=π\n(top)', ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.text(0, -1.3, 'θ=0\n(bottom)', ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax2.text(0, 1.3, 'θ=π\n(top)', ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Initialize beads
    bead1, = ax1.plot([], [], 'bo', markersize=12, label='Bead')
    bead2, = ax2.plot([], [], 'ro', markersize=12, label='Bead')
    
    # Trajectory trails
    trail1, = ax1.plot([], [], 'b-', alpha=0.6, linewidth=1)
    trail2, = ax2.plot([], [], 'r-', alpha=0.6, linewidth=1)
    
    # Phase space plot
    ax3.set_xlabel('Position θ (rad)')
    ax3.set_ylabel('Momentum p')
    ax3.set_title('Phase Space Comparison')
    ax3.grid(True)
    
    # Initialize phase space plots
    phase1, = ax3.plot([], [], 'b-', linewidth=2, label='Leapfrog')
    phase2, = ax3.plot([], [], 'r-', linewidth=2, label='Euler')
    ax3.legend()
    
    # Energy plot
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy vs Time')
    ax4.grid(True)
    
    # Calculate energies
    E_leap = hamiltonian(q_leap, p_leap, omega)
    E_euler = hamiltonian(q_euler, p_euler, omega)
    
    energy1, = ax4.plot([], [], 'b-', linewidth=2, label='Leapfrog')
    energy2, = ax4.plot([], [], 'r-', linewidth=2, label='Euler')
    ax4.legend()
    
    # Text for displaying current values
    text1 = ax1.text(-1.4, 1.3, '', fontsize=10)
    text2 = ax2.text(-1.4, 1.3, '', fontsize=10)
    
    # Animation parameters
    trail_length = min(200, len(t_leap) // 4)  # Length of trailing path
    
    def init():
        bead1.set_data([], [])
        bead2.set_data([], [])
        trail1.set_data([], [])
        trail2.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])
        energy1.set_data([], [])
        energy2.set_data([], [])
        text1.set_text('')
        text2.set_text('')
        return bead1, bead2, trail1, trail2, phase1, phase2, energy1, energy2, text1, text2
    
    def animate(frame):
        # Skip frames based on speed_factor
        i = int(frame * speed_factor)
        if i >= len(t_leap):
            i = len(t_leap) - 1
        
        # Current positions
        theta_leap = q_leap[i]
        theta_euler = q_euler[i]
        
        # Convert to Cartesian coordinates (bead position on hoop)
        # θ=0 is at BOTTOM of hoop (stable position), θ increases counterclockwise
        x_leap = R * np.sin(theta_leap)
        y_leap = R * np.cos(theta_leap)  # Positive cos gives bottom at θ=0
        
        x_euler = R * np.sin(theta_euler)  
        y_euler = R * np.cos(theta_euler)
        
        # Update bead positions
        bead1.set_data([x_leap], [y_leap])
        bead2.set_data([x_euler], [y_euler])
        
        # Update trails
        start_idx = max(0, i - trail_length)
        x_trail_leap = R * np.sin(q_leap[start_idx:i+1])
        y_trail_leap = R * np.cos(q_leap[start_idx:i+1])  # Fixed: θ=0 at bottom
        x_trail_euler = R * np.sin(q_euler[start_idx:i+1])
        y_trail_euler = R * np.cos(q_euler[start_idx:i+1])  # Fixed: θ=0 at bottom
        
        trail1.set_data(x_trail_leap, y_trail_leap)
        trail2.set_data(x_trail_euler, y_trail_euler)
        
        # Update phase space
        phase1.set_data(q_leap[:i+1], p_leap[:i+1])
        phase2.set_data(q_euler[:i+1], p_euler[:i+1])
        
        # Update phase space limits
        all_q = np.concatenate([q_leap[:i+1], q_euler[:i+1]])
        all_p = np.concatenate([p_leap[:i+1], p_euler[:i+1]])
        if len(all_q) > 0:
            margin = 0.1
            ax3.set_xlim(np.min(all_q) - margin, np.max(all_q) + margin)
            ax3.set_ylim(np.min(all_p) - margin, np.max(all_p) + margin)
        
        # Update energy plots
        energy1.set_data(t_leap[:i+1], E_leap[:i+1])
        energy2.set_data(t_euler[:i+1], E_euler[:i+1])
        
        # Update energy plot limits
        all_E = np.concatenate([E_leap[:i+1], E_euler[:i+1]])
        if len(all_E) > 0:
            ax4.set_xlim(0, t_leap[i])
            margin = 0.1 * (np.max(all_E) - np.min(all_E))
            ax4.set_ylim(np.min(all_E) - margin, np.max(all_E) + margin)
        
        # Update text information
        text1.set_text(f't = {t_leap[i]:.2f}s\nθ = {theta_leap:.3f} rad\np = {p_leap[i]:.3f}\nE = {E_leap[i]:.4f}')
        text2.set_text(f't = {t_euler[i]:.2f}s\nθ = {theta_euler:.3f} rad\np = {p_euler[i]:.3f}\nE = {E_euler[i]:.4f}')
        
        return bead1, bead2, trail1, trail2, phase1, phase2, energy1, energy2, text1, text2
    
    # Create animation
    total_frames = len(t_leap) // speed_factor
    interval = max(1, int(50 / speed_factor))  # Adjust for smooth animation
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames,
                        interval=interval, blit=True, repeat=True)
    
    plt.suptitle(f'Animated Comparison: Bead on Spinning Hoop (ω = {omega})', fontsize=16)
    plt.tight_layout()
    
    return fig, anim

def simulate_and_plot(omega, dt, num_steps, q0, p0):
    """Run simulation and create comparative plots"""
    
    # Get dynamics functions
    q_dot, p_dot = bead_on_hoop_dynamics(omega)
    
    # Run simulations
    t_leap, q_leap, p_leap = leapfrog_history(q0, p0, dt, num_steps, q_dot, p_dot)
    t_euler, q_euler, p_euler = euler_history(q0, p0, dt, num_steps, q_dot, p_dot)
    
    # Calculate energy conservation
    E_leap = hamiltonian(q_leap, p_leap, omega)
    E_euler = hamiltonian(q_euler, p_euler, omega)
    E0 = hamiltonian(q0, p0, omega)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position vs time
    ax1.plot(t_leap, q_leap, 'b-', label='Leapfrog', linewidth=2)
    ax1.plot(t_euler, q_euler, 'r--', label='Euler', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position θ (rad)')
    ax1.set_title('Position vs Time')
    ax1.legend()
    ax1.grid(True)
    
    # Momentum vs time
    ax2.plot(t_leap, p_leap, 'b-', label='Leapfrog', linewidth=2)
    ax2.plot(t_euler, p_euler, 'r--', label='Euler', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Momentum p')
    ax2.set_title('Momentum vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Phase space plot
    ax3.plot(q_leap, p_leap, 'b-', label='Leapfrog', linewidth=2)
    ax3.plot(q_euler, p_euler, 'r--', label='Euler', linewidth=2)
    ax3.plot(q0, p0, 'go', markersize=8, label='Initial')
    ax3.set_xlabel('Position θ (rad)')
    ax3.set_ylabel('Momentum p')
    ax3.set_title('Phase Space Trajectory')
    ax3.legend()
    ax3.grid(True)
    
    # Energy conservation
    ax4.plot(t_leap, (E_leap - E0) / E0, 'b-', label='Leapfrog', linewidth=2)
    ax4.plot(t_euler, (E_euler - E0) / E0, 'r--', label='Euler', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Relative Energy Error')
    ax4.set_title('Energy Conservation')
    ax4.legend()
    ax4.grid(True)
    ax4.set_yscale('symlog', linthresh=1e-10)
    
    plt.tight_layout()
    plt.suptitle(f'Bead on Spinning Hoop (ω = {omega})', y=1.02, fontsize=14)
    
    return fig, (t_leap, q_leap, p_leap), (t_euler, q_euler, p_euler)

# Main simulation
if __name__ == "__main__":
    print("=" * 60)
    print("BEAD ON SPINNING HOOP - EQUILIBRIUM ANALYSIS")
    print("=" * 60)
    
    # First, let's analyze the effective potential for different omega values
    print("1. EFFECTIVE POTENTIAL ANALYSIS")
    print("-" * 30)
    
    omega_test_values = [0.5, 1.0, 1.5, 2.0]
    fig_potential = plot_effective_potential(omega_test_values)
    plt.show()
    
    # Analyze equilibria for different omega values
    print("\n2. EQUILIBRIUM ANALYSIS")
    print("-" * 30)
    
    for omega in omega_test_values:
        print(f"\nω = {omega}:")
        equilibria = find_equilibrium_angles(omega)
        for theta_eq, stability in equilibria:
            if abs(theta_eq) < 2*np.pi:  # Only show primary range
                print(f"  θ = {theta_eq:.3f} rad ({theta_eq*180/np.pi:.1f}°) - {stability}")
    
    print("\n" + "=" * 60)
    print("COORDINATE SYSTEM CLARIFICATION:")
    print("- θ = 0: Bottom of hoop (stable when ω = 0)")
    print("- θ = π: Top of hoop (always unstable)") 
    print("- θ = ±arccos(-1/ω²): Stable equilibria on the SIDES (when ω ≥ 1)")
    print("- For ω = 1.5: Stable equilibria at θ ≈ ±1.23 rad (≈ ±70.5° from bottom)")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("3. SIMULATION WITH EQUILIBRIUM CONDITIONS")
    print("=" * 60)
    
    # Choose parameters that give stable equilibrium
    omega = 1.5  # Must be ≥ 1 for stable equilibria away from θ=0
    dt = 0.01
    T = 15.0     # Longer time to see stability
    num_steps = int(T / dt)
    
    print(f"\nSelected parameters: ω = {omega}")
    
    # Find good initial conditions
    result = find_initial_conditions_for_equilibrium(omega, oscillation_amplitude=0.05)
    
    if result is not None:
        q0, p0, theta_eq = result
        print(f"Initial conditions: θ₀ = {q0:.3f} rad, p₀ = {p0:.3f}")
        print(f"This should oscillate around θ = {theta_eq:.3f} rad ({theta_eq*180/np.pi:.1f}°)")
        
        # Run simulation with equilibrium conditions
        fig, leapfrog_data, euler_data = simulate_and_plot(omega, dt, num_steps, q0, p0)
        plt.show()
        
        # Check if we stayed away from θ=0
        t_leap, q_leap, p_leap = leapfrog_data
        min_angle = np.min(np.abs(q_leap))
        print(f"\nClosest approach to θ=0: {min_angle:.3f} rad ({min_angle*180/np.pi:.1f}°)")
        
        if min_angle > 0.5:  # If we stayed well away from θ=0
            print("✓ Success! The bead stays in equilibrium region without crossing θ=0")
            
            # Create animation with these parameters
            print("\nCreating animation with equilibrium conditions...")
            anim_dt = 0.02
            anim_steps = int(8.0 / anim_dt)
            
            fig_anim, animation = animate_comparison(omega, anim_dt, anim_steps, q0, p0, speed_factor=3)
            plt.show()
        else:
            print("⚠ The bead still crosses θ=0. Try larger ω or smaller initial displacement.")
    
    print("\n" + "=" * 60)
    print("4. ALTERNATIVE PARAMETERS FOR DIFFERENT EQUILIBRIA")
    print("=" * 60)
    
    # Show different parameter sets
    alternative_params = [
        (2.0, 0.03),  # Higher omega, smaller oscillation
        (1.2, 0.08),  # Just above critical omega
        (3.0, 0.02),  # Very high omega, very small oscillation
    ]
    
    for omega_alt, amp in alternative_params:
        print(f"\nTrying ω = {omega_alt}, amplitude = {amp}:")
        result_alt = find_initial_conditions_for_equilibrium(omega_alt, oscillation_amplitude=amp)
        if result_alt:
            q0_alt, p0_alt, theta_eq_alt = result_alt
            print(f"  → θ₀ = {q0_alt:.3f} rad, p₀ = {p0_alt:.3f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- For ω ≥ 1: Stable equilibria exist at θ = ±arccos(-1/ω²)")
    print("- Start with small oscillations around these equilibria")
    print("- Higher ω → equilibria further from θ=0 → easier to avoid crossing")
    print("- Use the functions above to automatically find good parameters!")
    print("=" * 60)
    
    # Additional comparison for different ω values
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    omega_values = [0.5, 1.0, 1.5, 2.0]
    
    for i, omega_test in enumerate(omega_values):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        q_dot, p_dot = bead_on_hoop_dynamics(omega_test)
        t_leap, q_leap, p_leap = leapfrog_history(q0, p0, dt, num_steps//2, q_dot, p_dot)
        t_euler, q_euler, p_euler = euler_history(q0, p0, dt, num_steps//2, q_dot, p_dot)
        
        ax.plot(q_leap, p_leap, 'b-', label='Leapfrog', linewidth=2)
        ax.plot(q_euler, p_euler, 'r--', label='Euler', linewidth=2)
        ax.set_xlabel('Position θ')
        ax.set_ylabel('Momentum p')
        ax.set_title(f'ω = {omega_test}')
        ax.grid(True)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.suptitle('Phase Space for Different Angular Velocities', y=1.02)
    plt.show()