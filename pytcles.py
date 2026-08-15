import pygame
import pygame_gui
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

# --- STANDARD MODEL PARTICLES ---

PARTICLES = {
    "Electron" : {"m": 0.511, "q": -1, "s": 0.5, "c": (0.2, 0.6, 1.0)},
    "Proton":   {"m": 938.2, "q": 1,  "s": 0.5, "c": (1.0, 0.2, 0.2)},
    "Muon" :     {"m": 105.6, "q": -1, "s": 0.5, "c": (0.2, 1.0, 0.5)},
    "Photon" :   {"m": 0.01,  "q": 0,  "s": 1.0, "c": (1.0, 1.0, 0.6)},
    "Higgs" :    {"m": 125.1, "q": 0,  "s": 0.0, "c": (1.0, 1.0, 1.0)}
}

# --- NOVAS CONSTANTES GLOBAIS ---
B_FIELD = np.array([0, 0, 0.5], dtype=np.float32)  # Campo Magnético (Eixo Z)
E_FIELD = np.array([0, -0.1, 0], dtype=np.float32) # Campo Elétrico (Eixo Y)

class QuantumParticle:
    def __init__(self, name, pos, vel):
        p = PARTICLES[name]
        self.name, self.mass, self.charge, self.spin_val, self.color = name, p['m'], p['q'], p['s'], p['c']
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.spin_vec = np.random.normal(0, 1, 3)
        self.spin_vec /= np.linalg.norm(self.spin_vec)
        self.dots = np.zeros((1500, 3), dtype=np.float32)
        self.collapsed = False

    def update(self, dt, mode, time_val):
        if not self.collapsed:
            # Applying electromagnectic field logic and spin
            f_lorentz = self.charge * (E_FIELD + np.cross(self.vel, B_FIELD))
            k = 0.5
            acceleration = (f_lorentz - k * self.pos) / self.mass
            self.vel += acceleration * dt
            self.pos += self.vel * dt

            # (E = 1/2 mv^2)
            energy = 0.5 * self.mass * np.linalg.norm(self.vel)**2
            frequency = (energy + self.mass) * 0.5 
            phase = time_val * frequency
            
            # Wave amplitude / Spred
            spread = 0.4 + (2.0 / (self.mass + 0.1))
            
            # Generate points with interference pattern
            num_points = 1500
            raw_dots = np.random.normal(0, spread, (num_points, 3))
            
            # Density probability and Wave modulator
            dist_from_center = np.linalg.norm(raw_dots, axis=1)
            wave_modulator = np.abs(np.sin(dist_from_center * 3.0 - phase))
            
            # Making dots more centered accordding the wave
            self.dots = self.pos + raw_dots * wave_modulator[:, np.newaxis]
            
            # Visual spin
            if self.spin_val > 0:
                self.spin_vec = np.array([np.cos(time_val), np.sin(time_val), 0.5])
                self.dots += np.cross(self.dots - self.pos, self.spin_vec) * self.spin_val * 0.2
        else:
            self.dots = self.pos.reshape(1, 3)

def resolve_collision(p1, p2):
    rel_pos = p1.pos - p2.pos
    dist = np.linalg.norm(rel_pos)
    if dist < 1.5 and np.dot(p1.vel - p2.vel, rel_pos) < 0:
        normal = rel_pos / dist
        v1n, v2n = np.dot(p1.vel, normal), np.dot(p2.vel, normal)
        new_v1n = (v1n * (p1.mass - p2.mass) + 2 * p2.mass * v2n) / (p1.mass + p2.mass)
        new_v2n = (v2n * (p2.mass - p1.mass) + 2 * p1.mass * v1n) / (p1.mass + p2.mass)
        p1.vel += (new_v1n - v1n) * normal
        p2.vel += (new_v2n - v2n) * normal

def main():
    pygame.init()
    res = (1280, 720)
    pygame.display.set_mode(res, DOUBLEBUF | OPENGL)
    ui = pygame_gui.UIManager(res)

    # --- UI ELEMENTS ---
    mode_list = ["Free View", "Collision", "Double Slit", "Entanglement"]
    mode_menu = pygame_gui.elements.UIDropDownMenu(mode_list, "Free View", Rect(20, 20, 200, 30), ui)
    p1_menu = pygame_gui.elements.UIDropDownMenu(list(PARTICLES.keys()), "Electron", Rect(20, 60, 200, 30), ui)
    p2_menu = pygame_gui.elements.UIDropDownMenu(list(PARTICLES.keys()), "Proton", Rect(20, 100, 200, 30), ui)
    obs_btn = pygame_gui.elements.UIButton(Rect(20, 140, 200, 30), "Measure (Collapse)", ui)
    reset_btn = pygame_gui.elements.UIButton(Rect(20, 180, 200, 30), "Reset System", ui)

    def spawn():
        m = mode_menu.selected_option[0]
        if m == "Collision":
            return [QuantumParticle(p1_menu.selected_option[0], [-8, 0.2, 0], [4, 0, 0]),
                    QuantumParticle(p2_menu.selected_option[1], [8, -0.2, 0], [-4, 0, 0])]
        elif m == "Entanglement":
            p = [QuantumParticle(p1_menu.selected_option[0], [-5, 0, 0], [0, 0, 0]),
                 QuantumParticle(p1_menu.selected_option[1], [5, 0, 0], [0, 0, 0])]
            p[1].color = p[0].color # Forçar mesma cor/estado
            return p
        else:
            return [QuantumParticle(p1_menu.selected_option[0], [0, 0, 0], [0, 0, 0])]

    particles = spawn()
    clock = pygame.time.Clock()
    zoom, rx, ry, time_val = -25.0, 0, 0, 0
    dragging = False

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
        for p in particles: p.update(dt, mode, time_val)
        if mode == "Collision" and len(particles) == 2:
            resolve_collision(particles[0], particles[1])
        if mode == "Entanglement" and particles[0].collapsed:
            particles[1].collapsed = True

        # --- RENDER ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Building Simulation
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, res[0]/res[1], 0.1, 1000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, zoom)
        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)

        glEnableClientState(GL_VERTEX_ARRAY)
        for p in particles:
            glColor3f(*p.color)
            glVertexPointer(3, GL_FLOAT, 0, p.dots)
            glDrawArrays(GL_POINTS, 0, len(p.dots))
        glDisableClientState(GL_VERTEX_ARRAY)

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

        pygame.display.flip()
if __name__ == "__main__":
    main()