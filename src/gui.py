#!/usr/bin/env python3
"""
GUI module for labeling training images in the Sand Mining Detection Tool.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

from src import config
from src.utils import load_labels, save_labels

class LabelingGUI:
    def __init__(self):
        self.images_folder = config.TRAINING_IMAGES_DIR
        self.labels_file = config.LABELS_FILE

        try:
            self.image_files = sorted([f for f in os.listdir(self.images_folder) if f.lower().endswith('.png')])
        except FileNotFoundError:
            messagebox.showerror("Error", f"Training image folder not found:\n{self.images_folder}")
            self.root = None
            return

        if not self.image_files:
            messagebox.showerror("Error", f"No images (.png) found in the training folder:\n{self.images_folder}\nCannot start labeling.")
            self.root = None  # Flag that GUI couldn't start
            return

        self.current_index = 0
        self.labels = load_labels()

        # Initialize missing labels for images currently in the folder
        for img_file in self.image_files:
            if img_file not in self.labels:
                self.labels[img_file] = -1  # -1 indicates unlabeled

        self.root = tk.Tk()
        self.root.title("Sand Mining Image Labeling Tool")
        self.root.geometry("1000x800")

        # Top Frame: Info and Progress
        self.info_frame = tk.Frame(self.root, pady=10)
        self.info_frame.pack(fill="x")
        self.progress_label = tk.Label(self.info_frame, text="", font=("Arial", 14))
        self.progress_label.pack()

        # Middle Frame: Image Canvas
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True)
        # Add binding to resize image when canvas size changes
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.current_image_object = None  # Store the loaded Image object
        self._image_id_on_canvas = None  # Store canvas image ID

        # Bottom Frame: Buttons
        self.button_frame = tk.Frame(self.root, pady=10)
        self.button_frame.pack(fill="x")

        self.prev_btn = tk.Button(self.button_frame, text="<< Previous", command=self.prev_image, width=15)
        self.prev_btn.pack(side=tk.LEFT, padx=20)

        self.no_mining_btn = tk.Button(self.button_frame, text="No Sand Mining (0)", command=lambda: self.set_label(0), width=20, height=2, bg="lightgreen")
        self.no_mining_btn.pack(side=tk.LEFT, padx=10)

        self.mining_btn = tk.Button(self.button_frame, text="Sand Mining (1)", command=lambda: self.set_label(1), width=20, height=2, bg="salmon")
        self.mining_btn.pack(side=tk.LEFT, padx=10)

        self.skip_btn = tk.Button(self.button_frame, text="Skip/Unlabel (?)", command=lambda: self.set_label(-1), width=15)
        self.skip_btn.pack(side=tk.LEFT, padx=10)

        self.next_btn = tk.Button(self.button_frame, text="Next >>", command=self.next_image, width=15)
        self.next_btn.pack(side=tk.RIGHT, padx=20)

        self.status_label = tk.Label(self.root, text="", font=("Arial", 12), fg="blue", anchor='w')
        self.status_label.pack(side=tk.BOTTOM, fill="x", padx=10, pady=5)

        # Help Button
        self.help_btn = tk.Button(self.info_frame, text="Help", command=self.show_help, width=8)
        self.help_btn.pack(side=tk.RIGHT, padx=20)

        # Bind keys
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("0", lambda e: self.set_label(0))
        self.root.bind("1", lambda e: self.set_label(1))
        self.root.bind("?", lambda e: self.set_label(-1))
        self.root.bind("<Escape>", lambda e: self.save_and_exit())
        self.root.bind("h", lambda e: self.show_help())

        # Delay initial image load slightly to allow canvas to get size
        self.root.after(100, self.load_image)
        self.root.protocol("WM_DELETE_WINDOW", self.save_and_exit)  # Save on close

    def show_help(self):
        """Show help information for labeling."""
        help_text = """
        Sand Mining Image Labeling Tool Help
        
        Keyboard Shortcuts:
        - Right Arrow: Next image
        - Left Arrow: Previous image
        - 0: Label as "No Sand Mining"
        - 1: Label as "Sand Mining"
        - ?: Skip/Unlabel image
        - ESC: Save and exit
        - h: Show this help
        
        Labeling Guidelines:
        - Label as "Sand Mining" if you see:
            * Visible extraction activity
            * Heavy equipment near water
            * Barges or boats carrying sand
            * Disturbed riverbanks with bare soil
            * Unnatural water color changes near banks
        
        - Label as "No Sand Mining" if:
            * Natural undisturbed riverbanks
            * No visible extraction activity
            * No heavy equipment
            * Clear water with natural patterns
        
        - Use Skip if:
            * Image is unclear (clouds, shadows)
            * You're uncertain about the label
            * Image has technical problems
        
        Your labels are used to train the sand mining detection model.
        Quality labeling is essential for accurate results!
        """
        messagebox.showinfo("Labeling Help", help_text)

    def on_canvas_resize(self, event):
        """Rescale and redraw the image when the canvas size changes."""
        if hasattr(self, '_resize_job'):
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self.display_image)

    def load_image(self):
        """Loads the image file but doesn't display it yet."""
        if not self.image_files:
            return
        
        self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
        img_file = self.image_files[self.current_index]
        img_path = os.path.join(self.images_folder, img_file)

        try:
            # Only load the image object here
            self.current_image_object = Image.open(img_path)
            self.display_image()  # Call display function
            self.update_status()

        except FileNotFoundError:
            messagebox.showerror("Error Loading Image", f"Image file not found: {img_path}\nIt might have been moved or deleted.")
            self.canvas.delete("all")
            self.current_image_object = None
            self.status_label.config(text=f"Error loading image: {img_file}", fg="red")
        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Could not load image file: {img_path}\n{type(e).__name__}: {e}")
            self.canvas.delete("all")  # Clear canvas on error
            self.current_image_object = None
            self.status_label.config(text=f"Error loading image: {img_file}", fg="red")

    def display_image(self):
        """Resizes and displays the currently loaded image object on the canvas."""
        if self.current_image_object is None:
            if self._image_id_on_canvas:
                self.canvas.delete(self._image_id_on_canvas)  # Clear previous image if load failed
            return

        try:
            img = self.current_image_object  # Use the loaded image

            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width < 2 or canvas_height < 2:  # Canvas might not be ready or too small
                return

            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:  # Image wider than canvas
                new_width = canvas_width
                new_height = int(new_width / img_ratio)
            else:  # Image taller than canvas
                new_height = canvas_height
                new_width = int(new_height * img_ratio)

            # Ensure dimensions are at least 1, prevent zero dimension errors
            new_width = max(1, new_width - 10)  # Add padding
            new_height = max(1, new_height - 10)  # Add padding

            # Use LANCZOS for better quality on downscaling
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Keep reference to PhotoImage to prevent garbage collection
            self._photo_ref = ImageTk.PhotoImage(img_resized)

            # Delete *only* the previous image item, not "all"
            if self._image_id_on_canvas:
                self.canvas.delete(self._image_id_on_canvas)
            # Center the image
            self._image_id_on_canvas = self.canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                anchor=tk.CENTER, 
                image=self._photo_ref
            )

        except Exception as e:
            # Handle potential errors during resizing/display
            print(f"Error displaying image: {e}")
            if self._image_id_on_canvas:
                self.canvas.delete(self._image_id_on_canvas)
            self.status_label.config(text=f"Error displaying image", fg="red")

    def update_status(self):
        """Update status and progress information."""
        if not self.image_files:
            return
        
        img_file = self.image_files[self.current_index]
        label = self.labels.get(img_file, -1)
        label_text = {0: "No Sand Mining", 1: "Sand Mining", -1: "Unlabeled"}[label]
        label_color = {0: "dark green", 1: "red", -1: "dark grey"}[label]  # Use darker colors for text

        total_images = len(self.image_files)
        labeled_count = sum(1 for lbl in self.labels.values() if lbl != -1)

        # Truncate long filenames in label
        display_filename = img_file if len(img_file) < 50 else img_file[:25] + "..." + img_file[-20:]
        self.progress_label.config(
            text=f"Image {self.current_index + 1} of {total_images} ({labeled_count} Labeled) - File: {display_filename}"
        )
        self.status_label.config(text=f"Current Label: {label_text}", fg=label_color)

        # Update button states (visually indicate current label)
        self.no_mining_btn.config(relief=tk.RAISED)
        self.mining_btn.config(relief=tk.RAISED)
        self.skip_btn.config(relief=tk.RAISED)
        
        if label == 0:
            self.no_mining_btn.config(relief=tk.SUNKEN)
        elif label == 1:
            self.mining_btn.config(relief=tk.SUNKEN)
        elif label == -1:
            self.skip_btn.config(relief=tk.SUNKEN)

    def set_label(self, label):
        """Set the label for the current image."""
        if not self.image_files:
            return
        
        img_file = self.image_files[self.current_index]
        self.labels[img_file] = label
        self.update_status()
        # Automatically move to next image after labeling
        self.next_image()  # Go to next immediately

    def next_image(self):
        """Move to the next image."""
        if not self.image_files:
            return
        
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()  # Load the new image

    def prev_image(self):
        """Move to the previous image."""
        if not self.image_files:
            return
        
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()  # Load the new image

    def save_and_exit(self):
        """Save labels and exit the application."""
        if self.root is None:  # Check if GUI initialization failed
            return
        
        if save_labels(self.labels):
            self.root.destroy()
        else:
            # Ask user if they want to exit anyway without saving
            if messagebox.askyesno("Exit Confirmation", "Failed to save labels. Do you still want to exit? (Changes will be lost)"):
                self.root.destroy()

def start_labeling_gui():
    """Start the labeling GUI."""
    # Check if images were actually downloaded
    try:
        image_files_exist = any(
            f.lower().endswith('.png') for f in os.listdir(config.TRAINING_IMAGES_DIR)
        )
    except FileNotFoundError:
        image_files_exist = False  # Folder doesn't exist yet

    if not image_files_exist:
        messagebox.showerror(
            "Labeling Error", 
            f"No training images (.png) found in:\n{config.TRAINING_IMAGES_DIR}\n\n"
            "Please ensure images were downloaded successfully before labeling."
        )
        return  # Don't start GUI if no images

    print("\nStarting Image Labeling GUI...")
    print("Instructions:")
    print("  - Use buttons or keys (0=No Mining, 1=Mining, ?=Skip) to label.")
    print("  - Use Left/Right arrow keys or buttons to navigate.")
    print("  - Press ESC or close window to save and exit.")
    print("  - Press 'h' for help and labeling guidelines.")
    
    try:
        gui = LabelingGUI()
        # If GUI failed to initialize (e.g., no images found), gui.root will be None
        if gui.root is None:
            print("GUI could not be initialized.")
        else:
            gui.root.mainloop()
    except tk.TclError as e:
        print(f"\nError initializing Tkinter GUI: {e}")
        print("This might happen if you are running in an environment without a display (e.g., a remote server via SSH without X forwarding).")
        print("Labeling requires a graphical interface.")
        
        # Offer alternative for headless environments
        print("\nAlternative: Use CSV file labeling in headless environments:")
        print(f"1. Create a CSV file at: {config.LABELS_FILE}")
        print("2. Format: 'filename,label' (header row + one row per image)")
        print("3. Labels: 0=No Sand Mining, 1=Sand Mining, -1=Unlabeled/Skip")