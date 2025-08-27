import pygame
import cv2
import torch
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# Initialize Pygame with double buffer for smoother animation
pygame.init()
flags = pygame.DOUBLEBUF | pygame.HWSURFACE
screen_width, screen_height = 1000, 1030
screen = pygame.display.set_mode((screen_width, screen_height), flags)
pygame.display.set_caption("Enhanced Traffic Signal Visualization")

# Realistic color palette
ROAD_COLOR = (60, 60, 60)
LANE_COLOR = (200, 200, 200)
CROSSWALK_COLOR = (250, 250, 250)
ASPHALT_TEXTURE = [
    (55, 55, 55), (65, 65, 65), (50, 50, 50), (70, 70, 70)
]
WHITE = (255, 255, 255)
RED = (255, 50, 50)
YELLOW = (255, 200, 0)
GREEN = (50, 255, 50)
GRAY = (30, 30, 30)
BLACK = (0, 0, 0)
PANEL_COLOR = (40, 40, 40)

# Improved fonts
font_large = pygame.font.Font(None, 36)
font_medium = pygame.font.Font(None, 28)
font_small = pygame.font.Font(None, 20)

# Load YOLO model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit()

def calculate_timers(car_counts):
    time_per_car = 1.5  # Reduced from 3 seconds
    min_time = 5.0      # Minimum time per lane
    max_time = 20.0     # Maximum time per lane
    
    # Calculate timers based on car counts
    timers = []
    for count in car_counts:
        timer = max(min_time, min(max_time, count * time_per_car))
        timers.append(timer)
    
    return timers

def count_vehicles(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    vehicle_classes = [2, 3, 5, 7]
    detections = results.xyxy[0].cpu().numpy()
    return sum(1 for det in detections if int(det[5]) in vehicle_classes)

def capture_from_file_dialog():
    Tk().withdraw()
    image_paths = askopenfilenames(title="Select Four Image Files",
                                 filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp")])
    if len(image_paths) != 4:
        print("Error: Please select exactly four images.")
        return [0, 0, 0, 0]

    vehicle_counts = []
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image at {image_path}.")
            vehicle_counts.append(0)
        else:
            vehicle_counts.append(count_vehicles(image))
            print(f"Image {i+1}: Detected {vehicle_counts[-1]} vehicles.")
    return vehicle_counts

def capture_one_by_one_images():
    cap = cv2.VideoCapture(0)
    directions = ["North", "East", "South", "West"]
    vehicle_counts = []

    print("Using webcam to capture images one by one.")
    for direction in directions:
        print(f"Position the camera for the {direction} side and press 's' to capture.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame for {direction} side. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                return [0, 0, 0, 0]

            cv2.imshow(f"Capture {direction} Side", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                vehicle_counts.append(count_vehicles(frame))
                print(f"Captured {direction} side. Detected {vehicle_counts[-1]} vehicles.")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return [0, 0, 0, 0]

    cap.release()
    cv2.destroyAllWindows()
    return vehicle_counts

def draw_intersection():
    # Textured main roads
    # Vertical road (north-south)
    for y in range(0, 1000, 5):
        for x in range(350, 650, 5):
            color = ASPHALT_TEXTURE[(x+y)//10 % len(ASPHALT_TEXTURE)]
            pygame.draw.rect(screen, color, (x, y, 5, 5))
    # Horizontal road (east-west)        
    for y in range(350, 650, 5):
        for x in range(0, 1000, 5):
            color = ASPHALT_TEXTURE[(x+y)//10 % len(ASPHALT_TEXTURE)]
            pygame.draw.rect(screen, color, (x, y, 5, 5))
    
    # Lane markings
    # North lane markings
    for y in range(0, 350, 50):
        pygame.draw.line(screen, LANE_COLOR, (500, y), (500, y+30), 3)
    # South lane markings
    for y in range(650, 1000, 50):
        pygame.draw.line(screen, LANE_COLOR, (500, y), (500, y+30), 3)
    for x in range(0, 350, 50):
        pygame.draw.line(screen, LANE_COLOR, (x, 500), (x+30, 500), 3)
    for x in range(650, 1000, 50):
        pygame.draw.line(screen, LANE_COLOR, (x, 500), (x+30, 500), 3)
    
    # Crosswalks
    for i in range(5):
        pygame.draw.rect(screen, CROSSWALK_COLOR, (450+i*20, 350, 10, 300))
        pygame.draw.rect(screen, CROSSWALK_COLOR, (350, 450+i*20, 300, 10))
    
    # Lane direction arrows
    arrow_n = [(500, 100), (480, 130), (520, 130)]
    arrow_s = [(500, 700), (480, 670), (520, 670)]
    arrow_e = [(900, 500), (870, 480), (870, 520)]
    arrow_w = [(100, 500), (130, 480), (130, 520)]
    pygame.draw.polygon(screen, WHITE, arrow_n)
    pygame.draw.polygon(screen, WHITE, arrow_s)
    pygame.draw.polygon(screen, WHITE, arrow_e)
    pygame.draw.polygon(screen, WHITE, arrow_w)


def draw_signals(active_lane, car_counts, timers, timer_remaining, frame_count):
    signals = [
        (500, 250, active_lane, car_counts[0], timers[0], "North"),
        (700, 500, active_lane, car_counts[1], timers[1], "East"),
        (500, 750, active_lane, car_counts[2], timers[2], "South"),
        (300, 500, active_lane, car_counts[3], timers[3], "West")
    ]

    for idx, (x, y, _, count, total_time, direction) in enumerate(signals):
        # Modern signal housing with metal texture
        if direction == "South":
            pygame.draw.rect(screen, (80, 80, 80), (x-30, y-150, 60, 240), border_radius=10)
            pygame.draw.rect(screen, (120, 120, 120), (x-28, y-148, 56, 236), border_radius=8)
        else:
            pygame.draw.rect(screen, (80, 80, 80), (x-30, y-110, 60, 240), border_radius=10)
            pygame.draw.rect(screen, (120, 120, 120), (x-28, y-108, 56, 236), border_radius=8)
        
        # Add metal texture lines
        for i in range(3):
            pygame.draw.line(screen, (150,150,150), (x-25, y-100+i*80), (x+25, y-100+i*80), 2)
        
        # Static light colors - completely turn off inactive lights
        light_colors = [
            (255, 50, 50) if active_lane != idx else (0,0,0),  # Red off when active
            (255, 200, 0) if (active_lane == idx and timer_remaining <= 3) else (0,0,0),
            (50, 255, 50) if active_lane == idx and timer_remaining > 3 else (0,0,0)
        ]
        
        # Draw lights with advanced effects
        for i, color in enumerate(light_colors):
            # Main light with gradient
            pygame.draw.circle(screen, color, (x, y-80 + i*80), 20)
            
        
        # Information display
        info_bg = pygame.Rect(x-50, y+100, 100, 60)
        pygame.draw.rect(screen, PANEL_COLOR, info_bg, border_radius=5)
        pygame.draw.rect(screen, (100,100,100), info_bg, 2, border_radius=5)
        
        dir_text = font_small.render(direction, True, WHITE)
        count_text = font_medium.render(f"{count} cars", True, WHITE)
        time_text = font_small.render(f"{timer_remaining:.1f}s" if active_lane == idx else f"{total_time}s", True, WHITE)
        
        screen.blit(dir_text, (x - dir_text.get_width()//2, y+105))
        screen.blit(count_text, (x - count_text.get_width()//2, y+125))
        screen.blit(time_text, (x - time_text.get_width()//2, y+145))

def draw_stats_panel(cycles, total_vehicles):
    panel = pygame.Rect(20, 20, 300, 120)
    pygame.draw.rect(screen, PANEL_COLOR, panel, border_radius=10)
    pygame.draw.rect(screen, (100,100,100), panel, 2, border_radius=10)
    
    title = font_large.render("Traffic Statistics", True, WHITE)
    cycle_text = font_medium.render(f"Cycles completed: {cycles}", True, WHITE)
    vehicle_text = font_medium.render(f"Total vehicles: {total_vehicles}", True, WHITE)
    
    screen.blit(title, (40, 30))
    screen.blit(cycle_text, (40, 70))
    screen.blit(vehicle_text, (40, 100))

def main():
    clock = pygame.time.Clock()
    running = True
    cycles_completed = 0
    total_vehicles = 0
    frame_count = 0

    print("Choose input type:\n1. Webcam for one-by-one images\n2. File dialog to select images")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        car_counts = capture_one_by_one_images()
    elif choice == "2":
        car_counts = capture_from_file_dialog()
    else:
        print("Invalid choice.")
        return

    if car_counts == [0, 0, 0, 0]:
        print("Error: No valid input provided. Exiting.")
        return

    total_vehicles = sum(car_counts)
    timers = calculate_timers(car_counts)
    active_lane = 0
    timer_remaining = timers[active_lane]

    while running:
        screen.fill((0, 50, 0))  # Green background
        
        draw_intersection()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        timer_remaining -= 1 / 30
        if timer_remaining <= 0:
            active_lane = (active_lane + 1) % 4
            timer_remaining = timers[active_lane]
            if active_lane == 0:
                cycles_completed += 1

        draw_signals(active_lane, car_counts, timers, timer_remaining, frame_count)
        frame_count += 1
        draw_stats_panel(cycles_completed, total_vehicles)
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
