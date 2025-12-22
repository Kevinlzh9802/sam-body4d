#!/usr/bin/env python3
"""
Interactive tool to get bounding boxes from video frames.
This helps you quickly find coordinates for --boxes argument in infer_video.py

Usage:
    python tools/get_bbox_interactive.py --video path/to/video.mp4 [--frame 0]
    
Instructions:
    - Click and drag to draw bounding boxes around people
    - Press 'n' to move to next person
    - Press 'q' when done
    - Coordinates will be printed in format ready for infer_video.py
"""

import cv2
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BBoxSelector:
    def __init__(self, image, frame_idx=0):
        self.image = image.copy()
        self.display = image.copy()
        self.frame_idx = frame_idx
        self.bboxes = []
        self.current_bbox = None
        self.drawing = False
        self.start_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.display = self.image.copy()
                # Draw all completed boxes
                for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
                    cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(self.display, f"Person {i+1}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw current box
                cv2.rectangle(self.display, self.start_point, (x, y), (0, 255, 255), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = x, y
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            self.bboxes.append((x1, y1, x2, y2))
            self.display = self.image.copy()
            # Draw all boxes
            for i, (bx1, by1, bx2, by2) in enumerate(self.bboxes):
                cv2.rectangle(self.display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(self.display, f"Person {i+1}", (bx1, by1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Added box for person {len(self.bboxes)}: ({x1}, {y1}, {x2}, {y2})")
    
    def run(self):
        window_name = f"Draw Bounding Boxes (Frame {self.frame_idx})"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("INSTRUCTIONS:")
        print("  1. Click and drag to draw bounding boxes around people")
        print("  2. Press 'r' to reset/clear all boxes")
        print("  3. Press 'u' to undo last box")
        print("  4. Press 'q' when done")
        print("="*60 + "\n")
        
        while True:
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.bboxes = []
                self.display = self.image.copy()
                print("Reset all boxes")
            elif key == ord('u'):
                if self.bboxes:
                    self.bboxes.pop()
                    self.display = self.image.copy()
                    for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
                        cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(self.display, f"Person {i+1}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print("Undid last box")
        
        cv2.destroyAllWindows()
        return self.bboxes


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool to get bounding boxes from video frames"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use (default: 0)")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Read the specified frame
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {args.frame} from video")
        cap.release()
        return 1
    
    height, width = frame.shape[:2]
    print(f"Video: {args.video}")
    print(f"Frame {args.frame}: {width}x{height}")
    
    cap.release()
    
    # Run interactive bbox selector
    selector = BBoxSelector(frame, args.frame)
    bboxes = selector.run()
    
    if not bboxes:
        print("\nNo bounding boxes selected.")
        return 0
    
    # Generate command-line arguments
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"\nFound {len(bboxes)} person(s)\n")
    
    # Format for infer_video.py
    box_args = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        obj_id = i + 1
        box_str = f'"{obj_id},{args.frame},{x1},{y1},{x2},{y2}"'
        box_args.append(box_str)
        print(f"Person {obj_id}: box at ({x1}, {y1}) to ({x2}, {y2})")
    
    print("\n" + "-"*60)
    print("Copy-paste this command:")
    print("-"*60)
    print(f"\npython infer_video.py \\")
    print(f"    --video {args.video} \\")
    print(f"    --config configs/body4d.yaml \\")
    print(f"    --output results/my_output \\")
    print(f"    --boxes {' '.join(box_args)}")
    print()
    
    # Also save to file
    output_file = "bbox_commands.txt"
    with open(output_file, 'w') as f:
        f.write(f"# Generated from {args.video}, frame {args.frame}\n")
        f.write(f"# Video dimensions: {width}x{height}\n\n")
        f.write(f"python infer_video.py \\\n")
        f.write(f"    --video {args.video} \\\n")
        f.write(f"    --config configs/body4d.yaml \\\n")
        f.write(f"    --output results/my_output \\\n")
        f.write(f"    --boxes {' '.join(box_args)}\n")
    
    print(f"Command also saved to: {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

