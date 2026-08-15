import pygame
import pygame_gui
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from collections import deque

# --- PHYSICS CONSTANTS ---
C           = 39.9                                      # Speed of Light
K_COULOMB   = 15.0                                      # F = k*(q1*q2)/r^2
B_FIELD     = np.array([0, 0, 0.5], dtype=np.float32)   # Z
E_FIELD     = np.array([0, -0.1, 0], dtype=np.float32)  # Y
A0          = 0.529                                     # Bohr Radius
BOHR_VEL    = 2.18e6                                    # Reference Velocity Scale

# --- STANDARD MODEL PARTICLES ---
PARTICLES = { # Mass, Charge, Spin and Color
    "Electron"  : {"m": 0.511, "q": -1, "s": 0.5, "c": (0.2, 0.6, 1.0)},
    "Proton"    : {"m": 938.2, "q": 1,  "s": 0.5, "c": (1.0, 0.2, 0.2)},
    "Positron"  : {"m": 0.511, "q": 1,  "s": 0.5, "c": (1.0, 0.5, 0.8)},
    "Muon"      : {"m": 105.6, "q": -1, "s": 0.5, "c": (0.2, 1.0, 0.5)},
    "Photon"    : {"m": 0.01,  "q": 0,  "s": 1.0, "c": (1.0, 1.0, 0.6)},
    "Neutron"   : {"m": 939.5, "q": 0,  "s": 0.5, "c": (0.6, 0.6, 0.6)},
    "Higgs"     : {"m": 125.1, "q": 0,  "s": 0.0, "c": (1.0, 1.0, 1.0)}
}

# --- ATOMS ---

ATOMS = { # Protons, Neutrons and Electrons by layer
    "Hydrogen"  : (1, 0, [1]),
    "Helium"    : (2, 2, [2]),
    "Lithium"   : (3, 4, [2, 1])
}

# --- UI MODES ---
FREE_VIEW_MODE      = "Free View"
COLLISION_MODE      = "Collision"
DOUBLE_SLIT_MODE    = "Double Slit"
ENTANGLEMENT_MODE   = "Entanglement"
ATOM                = "Atom"

class QuantumParticle:
    def __init__(self, name, pos, vel):
        p = PARTICLES[name]
        self.name = name
        self.mass0 = p['m']
        self.charge = p['q']
        self.color = p['c']
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.dots = np.zeros((800, 3), dtype=np.float32)
        self.collapsed = False
        self.is_annihilated = False

    def update_physics(self, dt, mode, time_val):
        if self.is_annihilated or self.collapsed: return

        # --- REALATIVISTIC MASS ---
        v_mag = np.linalg.norm(self.vel)
        gamma = 1.0 / np.sqrt(1.0 - min(v_mag**2 / C**2, 0.99))
        m_rel = self.mass0 * gamma

        # --- MAGNETIC FIELDS (LORENTZ) ---
        # F = q(E + v x B)
        f_lorentz = self.charge * (E_FIELD + np.cross(self.vel, B_FIELD))
        
        # --- SECOND LAW OF NEWTON  ---
        # a = F/m)
        accel = f_lorentz / m_rel
        self.vel += accel * dt
        self.pos += self.vel * dt

        # --- HEISENBERG UNCERTAINTY ---
        # Δx ∝ 1/√(m_rel)
        uncertainty = 1.8 / (np.sqrt(m_rel) + 0.1)
        raw_dots = np.random.normal(0, uncertainty, (800, 3))
        
        # WAVE FUNCTION REPRESENTATION |Ψ|²
        phase = time_val * (m_rel * 0.2)
        dist = np.linalg.norm(raw_dots, axis=1)
        wave_mod = 0.4 + 0.6 * np.abs(np.sin(dist * 3.0 - phase))
        
        self.dots = self.pos + raw_dots * wave_mod[:, np.newaxis]

class Atom:
    def __init__(self, name, center_pos):        
        z, n_neutrons, shells = ATOMS[name]
        self.particles = []
        self.center = np.array(center_pos, dtype=np.float32)

        # Building Atom Core (Protons and Neutrons)
        for i in range(z):
            self.particles.append(QuantumParticle("Proton", self.center + np.random.normal(0, 0.1, 3), [0,0,0]))
        for i in range(n_neutrons):
            self.particles.append(QuantumParticle("Neutron", self.center + np.random.normal(0, 0.1, 3), [0,0,0]))

        # Generating orbiting eletrons
        for shell_idx, count in enumerate(shells):
            n = shell_idx + 1 # Nível quântico principal
            radius = A0 * (n**2) / z * 10 # Escala visual
            
            for i in range(count):
                angle = (2 * np.pi / count) * i
                pos = self.center + [radius * np.cos(angle), radius * np.sin(angle), 0]
                # Orbital velocity TODO Might check this math later
                v_mag = 5.0 / n 
                vel = [-v_mag * np.sin(angle), v_mag * np.cos(angle), 0]
                
                electron = QuantumParticle("Electron", pos, vel)
                # Appending eletron to atom
                electron.parent_atom = self
                electron.orbital_n = n
                self.particles.append(electron)

    def update(self, dt, time_val):
        for p in self.particles:
            if p.name == "Electron":
                # Simulating Centripedal Force
                r_vec = p.pos - self.center
                dist = np.linalg.norm(r_vec)
                if dist > 0:
                    # Simplefying Coulomb's law to keep orbit intect
                    accel_dir = -r_vec / dist
                    force = (K_COULOMB * 2) / (dist**2) 
                    p.vel += accel_dir * force * dt
            
            p.update_physics(dt, "Free View", time_val)

def resolve_interactions(particles, dt):
    # --- COULOMB SPREAD (INTERACTION BETWEEN PAIRS)  ---
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles):
            if i >= j or p1.is_annihilated or p2.is_annihilated: continue
            
            rel_pos = p1.pos - p2.pos
            dist = np.linalg.norm(rel_pos)
            dist = max(dist, 0.99)

            # Eletrostatic force
            force_mag = K_COULOMB * (p1.charge * p2.charge) / (dist**2)
            force_vec = (rel_pos / dist) * force_mag
            
            p1.vel += (force_vec / p1.mass0) * dt
            p2.vel -= (force_vec / p2.mass0) * dt

            # Anihilation representation logic
            if dist < 1.2 and (p1.charge + p2.charge == 0) and (p1.mass0 == p2.mass0):
                p1.is_annihilated = p2.is_annihilated = True

def resolve_collision(p1, p2):
    rel_pos = p1.pos - p2.pos
    dist = np.linalg.norm(rel_pos)
    
    collision_threshold = 1.5 

    if dist < collision_threshold:
        rel_vel = p1.vel - p2.vel
        
        if np.dot(rel_vel, rel_pos) < 0:
            # Anihilation representation logic
            # ex: Eletrons + Positrons anihilates themselves
            if (p1.charge + p2.charge == 0) and (p1.mass0 == p2.mass0) and (p1.name != p2.name):
                p1.is_annihilated = True
                p2.is_annihilated = True
                return

            # --- ELASTIC COLLISION ---
            mass_sum = p1.mass0 + p2.mass0
            dot_product = np.dot(rel_vel, rel_pos) / (dist**2)
            
            # Updating speed
            p1.vel -= (2 * p2.mass0 / mass_sum) * dot_product * rel_pos
            p2.vel += (2 * p1.mass0 / mass_sum) * dot_product * rel_pos
            
            if hasattr(p1, 'trigger_collision_effect'):
                p1.trigger_collision_effect()
                p2.trigger_collision_effect()

def main():
    #pygame INIT
    pygame.init()
    res = (1280, 720)
    pygame.display.set_mode(res, DOUBLEBUF | OPENGL)
    ui = pygame_gui.UIManager(res)

    # --- UI ELEMENTS ---
    mode_list = [FREE_VIEW_MODE, COLLISION_MODE, DOUBLE_SLIT_MODE, ENTANGLEMENT_MODE, ATOM]
    mode_menu = pygame_gui.elements.UIDropDownMenu(mode_list, FREE_VIEW_MODE, Rect(20, 20, 200, 30), ui)
    p1_menu = pygame_gui.elements.UIDropDownMenu(list(PARTICLES.keys()), "Electron", Rect(20, 60, 200, 30), ui)
    p2_menu = pygame_gui.elements.UIDropDownMenu(list(PARTICLES.keys()), "Proton", Rect(20, 100, 200, 30), ui)
    obs_btn = pygame_gui.elements.UIButton(Rect(20, 140, 200, 30), "Measure (Collapse)", ui)
    reset_btn = pygame_gui.elements.UIButton(Rect(20, 180, 200, 30), "Reset System", ui)
    
    atoms = []
    
    def spawn():
        m = mode_menu.selected_option[0]
        if m == ATOM:
            atoms = [Atom("Helium", [0, 0, 0])]
            if len(atoms) > 0:
                if len(particles) > 0:
                    return atoms[0].particles
        if m == COLLISION_MODE:
            return [QuantumParticle(p1_menu.selected_option[0], [-14, 0, 0], [32, 0, 0]),
                    QuantumParticle(p2_menu.selected_option[1], [14, 0.1, 0], [-32, 0, 0])]
        return [QuantumParticle(p1_menu.selected_option[0], [0, 0, 0], [0, 0, 0])]

    particles = spawn()
    clock = pygame.time.Clock()
    zoom, rx, ry, time_val = -25.0, 0, 0, 0
    dragging = False

    for a in atoms:
        particles.extend(a.particles)

    ui_tex = glGenTextures(1)
        
    while True:
        dt = min(clock.tick(60) / 1000.0, 0.03)
        time_val += dt
        for event in pygame.event.get():
            if event.type == QUIT: return
            ui.process_events(event)
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == obs_btn:
                    for p in particles:
                        if not p.collapsed:
                            idx = np.random.randint(0, len(p.dots))
                            p.pos = p.dots[idx].copy()
                            p.dots = p.pos.reshape(1, 3)
                            p.collapsed = True
                if event.ui_element == reset_btn: particles = spawn()
            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED: particles = spawn()
            if event.type == MOUSEWHEEL: zoom += event.y
            if event.type == MOUSEBUTTONDOWN and event.button == 1: dragging = True
            if event.type == MOUSEBUTTONUP: dragging = False
            if event.type == MOUSEMOTION and dragging:
                dx, dy = event.rel
                ry += dx * 0.5; rx += dy * 0.5

        # --- PHYSICS ---
        mode = mode_menu.selected_option[0]
        for p in particles: p.update_physics(dt, mode, time_val)
        if mode == COLLISION_MODE and len(particles) == 2:
            resolve_collision(particles[0], particles[1])
        if mode == ENTANGLEMENT_MODE and particles[0].collapsed:
            particles[1].collapsed = True

        # --- RENDER ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Building Simulation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, res[0]/res[1], 0.01, 800.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, zoom)
        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)

        # --- POINT CONFIG ---
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        for p in particles:
            # DRAW WAVE FUNCTION CLOUD
            p_alpha = 0
            glPointSize(10.0)
            glColor4f(p.color[0], p.color[1], p.color[2], 0.2)
            
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, p.dots)
            glDrawArrays(GL_POINTS, 0, len(p.dots))
            glDisableClientState(GL_VERTEX_ARRAY)

        # --- ATOMS RENDERING ---
        if mode == ATOM and len(atoms) > 0:
            for a in atoms:
                a.update(dt, time_val)
                for p in a.particles:
                    if p.is_annihilated: continue

                    if p.name in ["Proton", "Neutron"]:
                        # Core
                        glPointSize(12.0 if p.name == "Proton" else 10.0)

                        draw_dots = [p.pos] 
                    else:
                        # Elétrons and Heisenberg probability cloud
                        glPointSize(4.0)
                        alpha = 0.3 * (1.0 / p.orbital_n)
                        draw_dots = p.dots

                    glColor4f(p.color[0], p.color[1], p.color[2], alpha)

                    glEnableClientState(GL_VERTEX_ARRAY)
                    v_data = np.array(draw_dots, dtype=np.float32)
                    glVertexPointer(3, GL_FLOAT, 0, v_data)
                    glDrawArrays(GL_POINTS, 0, len(v_data))
                    glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_POINT_SMOOTH)
        glDisable(GL_BLEND)

        # UI Surface Setup
        ui_surface = pygame.Surface(res, pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 0)) 
        ui.update(dt)
        ui.draw_ui(ui_surface)
        ui_data = pygame.image.tobytes(ui_surface, "RGBA", True)

        # ModelView and Projection Setup
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, res[0], res[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # UI Render Configuration
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, ui_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res[0], res[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, ui_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        # Draw UI Quad
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(res[0], 0)
        glTexCoord2f(1, 0); glVertex2f(res[0], res[1])
        glTexCoord2f(0, 0); glVertex2f(0, res[1])
        glEnd()
       
        # Clean Surface for next frame
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        #DISPLAY ON SURFACE
        pygame.display.flip()

if __name__ == "__main__":
    main()
