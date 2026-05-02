import cv2
import mss
import numpy as np

def main():
    capture_region = {"top": 40, "left": 0, "width": 1920, "height": 1080}
    
    print("="*50)
    print(" AI VISION CALIBRATOR ")
    print("="*50)
    print("1. A new window will pop up showing what the AI sees.")
    print("2. Drag your Trackmania window so the game fits perfectly inside.")
    print("3. Click on this preview window and press 'q' to close it.")
    print("="*50)

    with mss.mss() as sct:
        while True:
            # Grab the specific region of the screen
            screen = sct.grab(capture_region)
            img = np.array(screen)
            
            # Draw a green targeting crosshair in the center to help you align the car
            center_x, center_y = capture_region["width"] // 2, capture_region["height"] // 2
            cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
            cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
            
            # Add text so you know it's the AI view
            cv2.putText(img, "AI VISION FEED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(img, "Press 'q' to close", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display the live feed
            cv2.imshow("Trackmania Calibration Tool", img)

            # Press 'q' to quit the calibrator
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("✅ Calibration complete. You are ready to run train.py!")

if __name__ == "__main__":
    main()