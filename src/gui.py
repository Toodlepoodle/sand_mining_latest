#!/usr/bin/env python3
"""
Enhanced GUI for sand mining image labeling with mandatory region annotation.
- 500m buffer = tighter, closer images
- Draw boxes directly over mining areas
- Both image-level label AND region annotations feed into training
"""

import os
import sys
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np

from src import config
from src.utils import load_labels, save_labels


class LabelingGUI:
    def __init__(self):
        self.images_folder = config.TRAINING_IMAGES_DIR
        self.labels_file   = config.LABELS_FILE
        self.annotations_file = config.ANNOTATIONS_FILE

        try:
            self.image_files = sorted([
                f for f in os.listdir(self.images_folder)
                if f.lower().endswith('.png')
            ])
        except FileNotFoundError:
            messagebox.showerror("Error", f"Training image folder not found:\n{self.images_folder}")
            self.root = None
            return

        if not self.image_files:
            messagebox.showerror("Error", f"No images found in:\n{self.images_folder}")
            self.root = None
            return

        self.current_index = 0
        self.labels = load_labels()
        self.annotations = self._load_annotations()

        for img_file in self.image_files:
            if img_file not in self.labels:
                self.labels[img_file] = -1

        # ── Drawing state ──────────────────────────────────────────────────
        self.drawing = False
        self.draw_start = None
        self.current_rect = None
        self.annotation_type = 'sand_mining'  # default draw type
        self.annotation_mode = True           # ON by default now

        self.current_image_object = None
        self._image_id_on_canvas  = None
        self._photo_ref = None

        # ── Root window ────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Sand Mining Labeling — Draw Mining Regions")
        self.root.geometry("1100x850")
        self.root.configure(bg='#1e1e1e')

        self._build_ui()
        self.root.after(100, self.load_image)
        self.root.protocol("WM_DELETE_WINDOW", self.save_and_exit)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        # Top info bar
        top = tk.Frame(self.root, bg='#1e1e1e', pady=6)
        top.pack(fill='x')

        self.progress_label = tk.Label(
            top, text="", font=("Arial", 13, 'bold'),
            bg='#1e1e1e', fg='white'
        )
        self.progress_label.pack(side='left', padx=15)

        self.mode_label = tk.Label(
            top, text="✏️  DRAW MODE ON",
            font=("Arial", 12, 'bold'), bg='#1e1e1e', fg='#00d4aa'
        )
        self.mode_label.pack(side='right', padx=15)

        tk.Button(
            top, text="Help (h)", command=self.show_help,
            bg='#333', fg='white', relief='flat', padx=8
        ).pack(side='right', padx=5)

        # Canvas
        canvas_frame = tk.Frame(self.root, bg='#1e1e1e')
        canvas_frame.pack(fill='both', expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(canvas_frame, bg='#333333', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>',       self.on_canvas_resize)
        self.canvas.bind('<ButtonPress-1>',   self.on_mouse_press)
        self.canvas.bind('<B1-Motion>',       self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)

        # Annotation type selector
        type_frame = tk.Frame(self.root, bg='#1e1e1e', pady=4)
        type_frame.pack(fill='x', padx=8)

        tk.Label(
            type_frame, text="Draw type:", bg='#1e1e1e', fg='#aaa',
            font=("Arial", 10)
        ).pack(side='left', padx=5)

        self.type_var = tk.StringVar(value='sand_mining')
        types = [
            ('🔴 Sand Mining',      'sand_mining',      '#ff4444'),
            ('🔵 Water Disturb.',   'water_disturbance','#4488ff'),
            ('🟡 Equipment',        'equipment',        '#ffcc00'),
            ('🟢 No Mining (clear)','no_mining',        '#44cc44'),
        ]
        for label, val, col in types:
            tk.Radiobutton(
                type_frame, text=label, variable=self.type_var, value=val,
                bg='#1e1e1e', fg=col, selectcolor='#333',
                activebackground='#1e1e1e', activeforeground=col,
                font=("Arial", 10, 'bold'), command=self._update_type
            ).pack(side='left', padx=6)

        tk.Button(
            type_frame, text="❌ Clear boxes",
            command=self.clear_annotations,
            bg='#550000', fg='white', relief='flat', padx=8
        ).pack(side='right', padx=8)

        # Label buttons
        btn_frame = tk.Frame(self.root, bg='#1e1e1e', pady=6)
        btn_frame.pack(fill='x', padx=8)

        self.prev_btn = tk.Button(
            btn_frame, text="◀ Prev", command=self.prev_image,
            bg='#333', fg='white', width=10, relief='flat'
        )
        self.prev_btn.pack(side='left', padx=6)

        self.no_mining_btn = tk.Button(
            btn_frame, text="0  No Mining",
            command=lambda: self.set_label(0),
            bg='#1a5e20', fg='white', width=18, height=2,
            font=("Arial", 11, 'bold'), relief='flat'
        )
        self.no_mining_btn.pack(side='left', padx=6)

        self.mining_btn = tk.Button(
            btn_frame, text="1  Sand Mining",
            command=lambda: self.set_label(1),
            bg='#7b0000', fg='white', width=18, height=2,
            font=("Arial", 11, 'bold'), relief='flat'
        )
        self.mining_btn.pack(side='left', padx=6)

        self.skip_btn = tk.Button(
            btn_frame, text="?  Skip",
            command=lambda: self.set_label(-1),
            bg='#333', fg='#aaa', width=10, relief='flat'
        )
        self.skip_btn.pack(side='left', padx=6)

        self.next_btn = tk.Button(
            btn_frame, text="Next ▶", command=self.next_image,
            bg='#333', fg='white', width=10, relief='flat'
        )
        self.next_btn.pack(side='right', padx=6)

        # Status bar
        self.status_label = tk.Label(
            self.root, text="", font=("Arial", 11),
            bg='#1e1e1e', fg='#aaaaaa', anchor='w'
        )
        self.status_label.pack(fill='x', padx=12, pady=3)

        # Key bindings
        self.root.bind('<Left>',   lambda e: self.prev_image())
        self.root.bind('<Right>',  lambda e: self.next_image())
        self.root.bind('0',        lambda e: self.set_label(0))
        self.root.bind('1',        lambda e: self.set_label(1))
        self.root.bind('?',        lambda e: self.set_label(-1))
        self.root.bind('<Escape>', lambda e: self.save_and_exit())
        self.root.bind('h',        lambda e: self.show_help())
        self.root.bind('c',        lambda e: self.clear_annotations())

    # ── Annotation persistence ─────────────────────────────────────────────

    def _load_annotations(self):
        try:
            if os.path.exists(self.annotations_file):
                with open(self.annotations_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_annotations(self):
        try:
            os.makedirs(os.path.dirname(self.annotations_file), exist_ok=True)
            with open(self.annotations_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False

    # ── Image loading / display ────────────────────────────────────────────

    def load_image(self):
        if not self.image_files:
            return
        self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
        img_file  = self.image_files[self.current_index]
        img_path  = os.path.join(self.images_folder, img_file)
        try:
            self.current_image_object = Image.open(img_path).convert('RGB')
            self.display_image()
            self.update_status()
        except Exception as e:
            self.status_label.config(text=f"Error loading: {img_file} — {e}", fg='red')
            self.current_image_object = None

    def display_image(self):
        if self.current_image_object is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        img = self.current_image_object
        ratio = min((cw - 10) / img.width, (ch - 10) / img.height)
        nw = max(1, int(img.width  * ratio))
        nh = max(1, int(img.height * ratio))

        self._scale  = ratio
        self._offset = ((cw - nw) // 2, (ch - nh) // 2)

        img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)

        # Draw existing annotation boxes on the image
        # Boxes stored in original image pixel coords — scale only (NO canvas offset)
        img_draw = img_resized.copy()
        draw = ImageDraw.Draw(img_draw, 'RGBA')
        img_file = self.image_files[self.current_index]
        for ann in self.annotations.get(img_file, []):
            s  = self._scale
            ib = ann['bbox']
            bx = [int(ib[0]*s), int(ib[1]*s), int(ib[2]*s), int(ib[3]*s)]
            bx[0] = max(0, min(nw-1, bx[0]))
            bx[1] = max(0, min(nh-1, bx[1]))
            bx[2] = max(0, min(nw-1, bx[2]))
            bx[3] = max(0, min(nh-1, bx[3]))
            col = self._type_color(ann['type'])
            draw.rectangle(bx, outline=col, width=3)
            tag_w = min(120, max(10, bx[2] - bx[0]))
            tag_y2 = min(nh-1, bx[1] + 18)
            draw.rectangle([bx[0], bx[1], bx[0]+tag_w, tag_y2],
                           fill=(*self._hex_to_rgb(col), 180))
            draw.text((bx[0]+4, bx[1]+2), ann['type'][:12], fill='white')

        self._photo_ref = ImageTk.PhotoImage(img_draw)
        if self._image_id_on_canvas:
            self.canvas.delete(self._image_id_on_canvas)
        self._image_id_on_canvas = self.canvas.create_image(
            cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo_ref
        )

    def _type_color(self, t):
        return {
            'sand_mining':       '#ff4444',
            'water_disturbance': '#4488ff',
            'equipment':         '#ffcc00',
            'no_mining':         '#44cc44',
        }.get(t, '#ffffff')

    def _hex_to_rgb(self, h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def on_canvas_resize(self, event):
        if hasattr(self, '_resize_job'):
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self.display_image)

    # ── Drawing logic ──────────────────────────────────────────────────────

    def _update_type(self):
        self.annotation_type = self.type_var.get()

    def on_mouse_press(self, event):
        # Only start drawing if scale/offset are initialised (image loaded)
        if not hasattr(self, '_scale') or not hasattr(self, '_offset'):
            return
        self.drawing   = True
        self.draw_start = (event.x, event.y)
        if self.current_rect:
            self.canvas.delete(self.current_rect)

    def on_mouse_drag(self, event):
        if not self.drawing or not self.draw_start:
            return
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        col = self._type_color(self.annotation_type)
        self.current_rect = self.canvas.create_rectangle(
            self.draw_start[0], self.draw_start[1],
            event.x, event.y,
            outline=col, width=2, dash=(4, 4)
        )

    def on_mouse_release(self, event):
        if not self.drawing or not self.draw_start:
            return
        self.drawing = False

        x1, y1 = self.draw_start
        x2, y2 = event.x, event.y

        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        # Ignore tiny accidental clicks
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return

        # Normalise coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Convert canvas coords → image pixel coords
        ox, oy = self._offset
        scale  = self._scale
        img_bbox = [
            int((x1 - ox) / scale),
            int((y1 - oy) / scale),
            int((x2 - ox) / scale),
            int((y2 - oy) / scale),
        ]
        # Clamp to image bounds
        iw, ih = self.current_image_object.size
        img_bbox[0] = max(0, img_bbox[0])
        img_bbox[1] = max(0, img_bbox[1])
        img_bbox[2] = min(iw, img_bbox[2])
        img_bbox[3] = min(ih, img_bbox[3])

        if img_bbox[2] <= img_bbox[0] or img_bbox[3] <= img_bbox[1]:
            return

        img_file = self.image_files[self.current_index]
        if img_file not in self.annotations:
            self.annotations[img_file] = []

        self.annotations[img_file].append({
            'type':        self.annotation_type,
            'bbox':        img_bbox,           # image pixel coords
            'canvas_bbox': [x1, y1, x2, y2],  # canvas coords for redraw
        })

        # Auto-set label if drawing sand_mining box and image is unlabeled
        if self.annotation_type == 'sand_mining' and self.labels.get(img_file, -1) == -1:
            self.labels[img_file] = 1

        self._save_annotations()
        self.display_image()
        self.update_status()

    def clear_annotations(self):
        img_file = self.image_files[self.current_index]
        self.annotations[img_file] = []
        self._save_annotations()
        self.display_image()
        self.update_status()

    # ── Labeling ───────────────────────────────────────────────────────────

    def set_label(self, label):
        if not self.image_files:
            return
        img_file = self.image_files[self.current_index]
        self.labels[img_file] = label
        self.update_status()
        self.next_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def update_status(self):
        if not self.image_files:
            return
        img_file = self.image_files[self.current_index]
        label    = self.labels.get(img_file, -1)
        n_boxes  = len(self.annotations.get(img_file, []))
        labeled  = sum(1 for v in self.labels.values() if v != -1)
        total    = len(self.image_files)

        label_text  = {0: "✅ No Mining", 1: "⛏️  Sand Mining", -1: "— Unlabeled"}[label]
        label_color = {0: '#44cc44',      1: '#ff4444',        -1: '#888888'}[label]

        short = img_file if len(img_file) < 50 else img_file[:25] + '...' + img_file[-20:]
        self.progress_label.config(
            text=f"Image {self.current_index + 1}/{total}  ({labeled} labeled)  |  {short}"
        )
        self.status_label.config(
            text=f"Label: {label_text}   |   Boxes drawn: {n_boxes}   |   "
                 f"Draw type: {self.annotation_type}   |   "
                 f"[0/1=label  ←/→=navigate  C=clear boxes  ESC=save+exit]",
            fg=label_color
        )

        # Button relief
        for btn, lbl in [(self.no_mining_btn, 0), (self.mining_btn, 1), (self.skip_btn, -1)]:
            btn.config(relief='sunken' if label == lbl else 'flat')

    # ── Help ───────────────────────────────────────────────────────────────

    def show_help(self):
        messagebox.showinfo("How to Label", """
WORKFLOW (do both steps for best results):

STEP 1 — Draw boxes on mining areas:
  • Click and drag directly on the satellite image
  • Draw a box tightly around the mining site
  • Use the radio buttons to select what you're drawing:
    🔴 Sand Mining  — active extraction pit or dredge
    🔵 Water Disturbance — turbid/disturbed water
    🟡 Equipment — machines, barges, trucks
    🟢 No Mining — clearly undisturbed area
  • Press C to clear all boxes on current image

STEP 2 — Set overall image label:
  • Press 1 → whole image has sand mining
  • Press 0 → whole image has no mining
  • Press ? → skip/uncertain
  • Arrow keys to navigate

WHAT TO LOOK FOR at 500m zoom:
  • Sandy bare patches cut into riverbank
  • Barges or boats near shore
  • Excavator tracks / disturbed soil
  • Discoloured (brown/turbid) water near bank
  • Straight-edged cuts in the bank line

TIPS:
  • Drawing a sand_mining box auto-sets label to 1
  • You can draw multiple boxes per image
  • More boxes = better spatial features for the model
  • ESC saves everything and exits
""")

    # ── Save and exit ──────────────────────────────────────────────────────

    def save_and_exit(self):
        if self.root is None:
            return
        self._save_annotations()
        if save_labels(self.labels):
            n_ann = sum(len(v) for v in self.annotations.values())
            print(f"\nSaved {sum(1 for v in self.labels.values() if v != -1)} labels "
                  f"and {n_ann} region annotations.")
            self.root.destroy()
        else:
            if messagebox.askyesno("Exit", "Failed to save labels. Exit anyway?"):
                self.root.destroy()


def start_labeling_gui():
    try:
        image_files_exist = any(
            f.lower().endswith('.png')
            for f in os.listdir(config.TRAINING_IMAGES_DIR)
        )
    except FileNotFoundError:
        image_files_exist = False

    if not image_files_exist:
        messagebox.showerror(
            "No Images",
            f"No training images found in:\n{config.TRAINING_IMAGES_DIR}\n\n"
            "Run --mode download first."
        )
        return

    print("\nStarting Labeling GUI...")
    print("  Draw boxes over mining areas, then press 1/0 to label.")
    print("  Press ESC or close window to save and exit.\n")

    try:
        gui = LabelingGUI()
        if gui.root is not None:
            gui.root.mainloop()
    except tk.TclError as e:
        print(f"\nCannot open GUI: {e}")
        print("Requires a display. On remote servers use X forwarding.")