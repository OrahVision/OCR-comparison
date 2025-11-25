"""
OCR Results Browser

A GUI application to browse images and their OCR results from different providers.
Features:
- Image display on left, OCR text on right
- Provider selector to switch between OCR results
- Read Aloud button for TTS
- Navigation through images
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import threading
from pathlib import Path
from PIL import Image, ImageTk

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.database import OCRDatabase

# Cloud/on-demand providers available for selective OCR
# These cost money or require GPU, so we run them on-demand
CLOUD_PROVIDERS = {
    # Cloud APIs
    'google_vision': 'Google Vision',
    'gemini_vision': 'Gemini Flash',
    'claude_vision': 'Claude Haiku',
    'openai_vision': 'GPT-4o Mini',
    'azure_doc_intel': 'Azure Doc Intel',
    'aws_textract': 'AWS Textract',
    # Local VLMs (require GPU)
    'deepseek_ocr': 'DeepSeek OCR',
    'florence2': 'Florence-2',
    'qwen2vl': 'Qwen2-VL',
}


class OCRBrowser(tk.Tk):
    """Main application window for browsing OCR results."""

    def __init__(self, db_path: str = "ocr_results.db"):
        super().__init__()

        self.title("OCR Results Browser")
        self.geometry("1400x900")
        self.minsize(1000, 600)

        # Database
        self.db_path = db_path
        self.db = None
        self.images = []
        self.current_index = 0
        self.current_image_id = None
        self.providers = []
        self.current_provider = None

        # Image display
        self.photo_image = None

        # TTS state
        self.tts_thread = None
        self.tts_stop_flag = False

        # Setup UI
        self._setup_ui()
        self._setup_bindings()

        # Load database
        self._load_database()

    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # === Top toolbar ===
        self.toolbar = ttk.Frame(self.main_frame)
        self.toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Navigation buttons
        self.btn_prev = ttk.Button(self.toolbar, text="< Previous", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.btn_next = ttk.Button(self.toolbar, text="Next >", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # Image counter
        self.lbl_counter = ttk.Label(self.toolbar, text="0 / 0")
        self.lbl_counter.pack(side=tk.LEFT, padx=20)

        # Jump to image
        ttk.Label(self.toolbar, text="Go to:").pack(side=tk.LEFT, padx=(20, 5))
        self.entry_goto = ttk.Entry(self.toolbar, width=6)
        self.entry_goto.pack(side=tk.LEFT, padx=2)
        self.btn_goto = ttk.Button(self.toolbar, text="Go", command=self.goto_image)
        self.btn_goto.pack(side=tk.LEFT, padx=2)

        # Export button
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.btn_export = ttk.Button(self.toolbar, text="Export Results", command=self.export_results)
        self.btn_export.pack(side=tk.LEFT, padx=2)

        # Provider selector
        ttk.Label(self.toolbar, text="Provider:").pack(side=tk.LEFT, padx=(30, 5))
        self.provider_var = tk.StringVar()
        self.combo_provider = ttk.Combobox(
            self.toolbar,
            textvariable=self.provider_var,
            state="readonly",
            width=15
        )
        self.combo_provider.pack(side=tk.LEFT, padx=2)
        self.combo_provider.bind("<<ComboboxSelected>>", self.on_provider_changed)

        # Read Aloud button
        self.btn_read = ttk.Button(
            self.toolbar,
            text="Read Aloud",
            command=self.read_aloud
        )
        self.btn_read.pack(side=tk.RIGHT, padx=2)

        self.btn_stop_read = ttk.Button(
            self.toolbar,
            text="Stop",
            command=self.stop_reading,
            state=tk.DISABLED
        )
        self.btn_stop_read.pack(side=tk.RIGHT, padx=2)

        # Separator
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Cloud OCR section
        self.btn_cloud_ocr = ttk.Button(
            self.toolbar,
            text="Run Cloud OCR",
            command=self.run_cloud_ocr
        )
        self.btn_cloud_ocr.pack(side=tk.RIGHT, padx=2)

        self.cloud_provider_var = tk.StringVar()
        self.combo_cloud = ttk.Combobox(
            self.toolbar,
            textvariable=self.cloud_provider_var,
            state="readonly",
            width=15,
            values=list(CLOUD_PROVIDERS.values())
        )
        self.combo_cloud.pack(side=tk.RIGHT, padx=2)
        if CLOUD_PROVIDERS:
            self.combo_cloud.set(list(CLOUD_PROVIDERS.values())[0])

        ttk.Label(self.toolbar, text="Cloud:").pack(side=tk.RIGHT, padx=(0, 5))

        # === Left panel: Image ===
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image")
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        # Canvas for image with scrollbars
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Image path label
        self.lbl_image_path = ttk.Label(self.image_frame, text="", wraplength=600)
        self.lbl_image_path.pack(side=tk.BOTTOM, pady=5)

        # === Right panel: OCR Text ===
        self.text_frame = ttk.LabelFrame(self.main_frame, text="OCR Result")
        self.text_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # Provider info
        self.info_frame = ttk.Frame(self.text_frame)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_provider_info = ttk.Label(
            self.info_frame,
            text="",
            font=("", 9)
        )
        self.lbl_provider_info.pack(side=tk.LEFT)

        # Text widget with scrollbar
        self.text_scroll = ttk.Scrollbar(self.text_frame)
        self.text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_widget = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            yscrollcommand=self.text_scroll.set,
            padx=10,
            pady=10
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_scroll.config(command=self.text_widget.yview)

        # === Status bar ===
        self.status_bar = ttk.Label(self.main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _setup_bindings(self):
        """Setup keyboard bindings."""
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Home>", lambda e: self.goto_first())
        self.bind("<End>", lambda e: self.goto_last())
        self.bind("<Return>", lambda e: self.goto_image())
        self.bind("<Escape>", lambda e: self.stop_reading())

    def _load_database(self):
        """Load database and populate image list."""
        if not os.path.exists(self.db_path):
            messagebox.showerror(
                "Database Not Found",
                f"Database not found: {self.db_path}\n\n"
                "Run the batch processor first to create OCR results."
            )
            return

        try:
            self.db = OCRDatabase(self.db_path)

            # Get all images with OCR results
            self.images = self.db.get_images_with_provider_count()

            # Get providers
            self.providers = self.db.get_providers()

            if not self.images:
                self.status_bar.config(text="No images with OCR results found in database")
                return

            if not self.providers:
                self.status_bar.config(text="No OCR results found in database")
                return

            # Setup provider combo
            self.combo_provider['values'] = self.providers
            self.combo_provider.set(self.providers[0])
            self.current_provider = self.providers[0]

            # Show first image
            self.current_index = 0
            self._display_current()

            self.status_bar.config(
                text=f"Loaded {len(self.images)} images, {len(self.providers)} providers"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {e}")

    def _display_current(self):
        """Display current image and OCR result."""
        if not self.images:
            return

        image_data = self.images[self.current_index]
        self.current_image_id = image_data['id']
        image_path = image_data['path']

        # Update counter
        self.lbl_counter.config(text=f"{self.current_index + 1} / {len(self.images)}")

        # Update path label
        self.lbl_image_path.config(text=image_path)

        # Load and display image
        self._display_image(image_path)

        # Load OCR result
        self._display_ocr_result()

    def _display_image(self, image_path: str):
        """Load and display image on canvas."""
        if not os.path.exists(image_path):
            self.canvas.delete("all")
            self.canvas.create_text(
                300, 300,
                text=f"Image not found:\n{image_path}",
                fill="red",
                font=("", 12)
            )
            return

        try:
            # Load image
            img = Image.open(image_path)

            # Get canvas size
            self.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width < 100:
                canvas_width = 600
            if canvas_height < 100:
                canvas_height = 700

            # Calculate scale to fit
            scale = min(
                canvas_width / img.width,
                canvas_height / img.height
            )

            # Only scale down, not up
            if scale < 1:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(img)

            # Clear canvas and display
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                anchor=tk.CENTER,
                image=self.photo_image
            )

        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                300, 300,
                text=f"Error loading image:\n{e}",
                fill="red",
                font=("", 12)
            )

    def _display_ocr_result(self):
        """Display OCR result for current image and provider."""
        if not self.current_image_id or not self.current_provider:
            return

        result = self.db.get_ocr_result(self.current_image_id, self.current_provider)

        # Update text widget
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)

        if result:
            if result['success']:
                self.text_widget.insert(tk.END, result['text'])
                self.lbl_provider_info.config(
                    text=f"Words: {result['word_count']} | "
                         f"Confidence: {result['confidence']:.1%} | "
                         f"Time: {result['processing_time_ms']:.0f}ms"
                )
            else:
                self.text_widget.insert(tk.END, f"OCR Failed: {result.get('error', 'Unknown error')}")
                self.lbl_provider_info.config(text="OCR Failed")
        else:
            self.text_widget.insert(tk.END, f"No OCR result for provider: {self.current_provider}")
            self.lbl_provider_info.config(text="No result")

        self.text_widget.config(state=tk.DISABLED)

    def prev_image(self):
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current()

    def next_image(self):
        """Go to next image."""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self._display_current()

    def goto_first(self):
        """Go to first image."""
        self.current_index = 0
        self._display_current()

    def goto_last(self):
        """Go to last image."""
        self.current_index = len(self.images) - 1
        self._display_current()

    def goto_image(self):
        """Go to specific image number."""
        try:
            num = int(self.entry_goto.get())
            if 1 <= num <= len(self.images):
                self.current_index = num - 1
                self._display_current()
            else:
                messagebox.showwarning("Invalid", f"Enter a number between 1 and {len(self.images)}")
        except ValueError:
            messagebox.showwarning("Invalid", "Please enter a valid number")

    def on_provider_changed(self, event=None):
        """Handle provider selection change."""
        self.current_provider = self.provider_var.get()
        self._display_ocr_result()

    def export_results(self):
        """Export OCR results to a text file."""
        if not self.images:
            messagebox.showwarning("No Data", "No images loaded to export")
            return

        # Ask user what to export
        choice = messagebox.askyesnocancel(
            "Export Options",
            "Export ALL images?\n\n"
            "Yes = Export all images\n"
            "No = Export current image only\n"
            "Cancel = Abort"
        )

        if choice is None:  # Cancel
            return

        export_all = choice

        # Ask for file location
        default_name = "ocr_results_all.txt" if export_all else f"ocr_result_{self.current_index + 1}.txt"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_name,
            title="Save OCR Results"
        )

        if not file_path:
            return

        try:
            self.status_bar.config(text="Exporting...")
            self.update_idletasks()

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("OCR RESULTS EXPORT\n")
                f.write("=" * 80 + "\n\n")

                images_to_export = self.images if export_all else [self.images[self.current_index]]

                for i, image_data in enumerate(images_to_export):
                    image_id = image_data['id']
                    image_path = image_data['path']

                    f.write("-" * 80 + "\n")
                    f.write(f"IMAGE {i + 1}: {image_path}\n")
                    f.write("-" * 80 + "\n\n")

                    # Get all OCR results for this image
                    results = self.db.get_all_ocr_results_for_image(image_id)

                    if not results:
                        f.write("  (No OCR results available)\n\n")
                        continue

                    for result in results:
                        provider = result['provider']
                        f.write(f">>> PROVIDER: {provider.upper()}\n")
                        f.write(f"    Confidence: {result['confidence']:.1%}\n")
                        f.write(f"    Words: {result['word_count']}\n")
                        f.write(f"    Processing time: {result['processing_time_ms']:.0f}ms\n")

                        if result['success']:
                            f.write(f"    Status: Success\n")
                            f.write("\n--- TEXT START ---\n")
                            f.write(result['text'])
                            f.write("\n--- TEXT END ---\n\n")
                        else:
                            f.write(f"    Status: FAILED - {result.get('error', 'Unknown error')}\n\n")

                f.write("=" * 80 + "\n")
                f.write("END OF EXPORT\n")
                f.write("=" * 80 + "\n")

            count = len(images_to_export)
            self.status_bar.config(text=f"Exported {count} image(s) to {os.path.basename(file_path)}")
            messagebox.showinfo("Export Complete", f"Exported {count} image(s) to:\n{file_path}")

        except Exception as e:
            self.status_bar.config(text="Export failed")
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def read_aloud(self):
        """Read current OCR text aloud using TTS."""
        # Get current text
        text = self.text_widget.get("1.0", tk.END).strip()

        if not text:
            messagebox.showwarning("No Text", "No text to read")
            return

        # Start TTS in background thread
        self.tts_stop_flag = False
        self.btn_read.config(state=tk.DISABLED)
        self.btn_stop_read.config(state=tk.NORMAL)
        self.status_bar.config(text="Reading aloud...")

        self.tts_thread = threading.Thread(target=self._do_tts, args=(text,))
        self.tts_thread.daemon = True
        self.tts_thread.start()

    def _do_tts(self, text: str):
        """Perform TTS in background thread."""
        try:
            # Import TTS components
            from core.tts_service import TTSService
            from core.audio_player import StreamingAudioPlayer, is_audio_available
            from api_integration.api_utils import get_user_credentials

            if not is_audio_available():
                self.after(0, lambda: messagebox.showerror("Error", "Audio playback not available"))
                return

            # Get credentials
            credentials = get_user_credentials()
            if not credentials:
                self.after(0, lambda: messagebox.showerror(
                    "Error",
                    "No credentials found. Please set up API credentials."
                ))
                return

            # Create TTS service
            service = TTSService(credentials, apply_cleanup=True)

            # Create audio player and stream
            with StreamingAudioPlayer() as player:
                def stop_check():
                    return self.tts_stop_flag

                service.synthesize_streaming_progressive(text, player, stop_callback=stop_check)

        except ImportError as e:
            self.after(0, lambda: messagebox.showerror(
                "Import Error",
                f"TTS components not available: {e}"
            ))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("TTS Error", f"TTS failed: {e}"))
        finally:
            self.after(0, self._tts_finished)

    def _tts_finished(self):
        """Called when TTS finishes."""
        self.btn_read.config(state=tk.NORMAL)
        self.btn_stop_read.config(state=tk.DISABLED)
        self.status_bar.config(text="Ready")

    def stop_reading(self):
        """Stop TTS playback."""
        self.tts_stop_flag = True
        self.status_bar.config(text="Stopping...")

    def run_cloud_ocr(self):
        """Run OCR on current image using selected cloud provider."""
        if not self.images or self.current_image_id is None:
            messagebox.showwarning("No Image", "No image selected")
            return

        # Get selected cloud provider
        display_name = self.cloud_provider_var.get()
        provider_key = None
        for key, name in CLOUD_PROVIDERS.items():
            if name == display_name:
                provider_key = key
                break

        if not provider_key:
            messagebox.showwarning("No Provider", "Please select a cloud provider")
            return

        # Get current image path
        image_data = self.images[self.current_index]
        image_path = image_data['path']

        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image not found: {image_path}")
            return

        # Check if already processed
        if self.db.has_ocr_result(self.current_image_id, provider_key):
            if not messagebox.askyesno(
                "Already Processed",
                f"This image already has OCR results from {display_name}.\n\n"
                "Do you want to run it again? (This will replace the existing result)"
            ):
                return

        # Disable button during processing
        self.btn_cloud_ocr.config(state=tk.DISABLED)
        self.status_bar.config(text=f"Running {display_name} OCR...")
        self.update_idletasks()

        # Run in background thread
        thread = threading.Thread(
            target=self._do_cloud_ocr,
            args=(image_path, provider_key, display_name)
        )
        thread.daemon = True
        thread.start()

    def _do_cloud_ocr(self, image_path: str, provider_key: str, display_name: str):
        """Perform cloud OCR in background thread."""
        import time
        import asyncio

        try:
            provider = None
            start_time = time.perf_counter()

            # Import and create the appropriate provider
            if provider_key == 'google_vision':
                from ocr.providers.google_vision import GoogleVisionOCR, check_google_cloud_auth
                if not check_google_cloud_auth():
                    self.after(0, lambda: messagebox.showerror(
                        "Authentication Error",
                        "Google Cloud credentials not found.\n\n"
                        "Set up via GOOGLE_APPLICATION_CREDENTIALS or gcloud auth."
                    ))
                    return
                provider = GoogleVisionOCR()

            elif provider_key == 'gemini_vision':
                from ocr.providers.gemini_vision import GeminiVisionOCR, get_gemini_api_key
                if not get_gemini_api_key():
                    self.after(0, lambda: messagebox.showerror(
                        "Authentication Error",
                        "Gemini API key not found.\n\n"
                        "Set via keyring (gemini/api_key) or GEMINI_API_KEY env var."
                    ))
                    return
                provider = GeminiVisionOCR()

            elif provider_key == 'claude_vision':
                from ocr.providers.claude_vision import ClaudeVisionOCR, get_anthropic_api_key
                if not get_anthropic_api_key():
                    self.after(0, lambda: messagebox.showerror(
                        "Authentication Error",
                        "Anthropic API key not found.\n\n"
                        "Set via keyring (anthropic/api_key) or ANTHROPIC_API_KEY env var."
                    ))
                    return
                provider = ClaudeVisionOCR()

            elif provider_key == 'openai_vision':
                from ocr.providers.openai_vision import OpenAIVisionOCR, get_openai_api_key
                if not get_openai_api_key():
                    self.after(0, lambda: messagebox.showerror(
                        "Authentication Error",
                        "OpenAI API key not found.\n\n"
                        "Set via keyring (openai/api_key) or OPENAI_API_KEY env var."
                    ))
                    return
                provider = OpenAIVisionOCR()

            elif provider_key == 'azure_doc_intel':
                from ocr.providers.azure_doc_intel import AzureDocIntelOCR, get_azure_credentials
                endpoint, key = get_azure_credentials()
                if not endpoint or not key:
                    self.after(0, lambda: messagebox.showerror(
                        "Authentication Error",
                        "Azure credentials not found.\n\n"
                        "Set via AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and _KEY env vars."
                    ))
                    return
                provider = AzureDocIntelOCR()

            elif provider_key == 'aws_textract':
                from ocr.providers.aws_textract import AWSTextractOCR, check_aws_credentials
                if not check_aws_credentials():
                    self.after(0, lambda: messagebox.showerror(
                        "Authentication Error",
                        "AWS credentials not found.\n\n"
                        "Set via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars,\n"
                        "or configure via: aws configure"
                    ))
                    return
                provider = AWSTextractOCR()

            elif provider_key == 'deepseek_ocr':
                from ocr.providers.deepseek_ocr import DeepSeekOCRProvider, DEEPSEEK_AVAILABLE
                if not DEEPSEEK_AVAILABLE:
                    self.after(0, lambda: messagebox.showerror(
                        "Not Available",
                        "DeepSeek OCR requires GPU with 16GB+ VRAM.\n\n"
                        "Install: pip install transformers torch"
                    ))
                    return
                provider = DeepSeekOCRProvider()

            elif provider_key == 'florence2':
                from ocr.providers.florence2_provider import Florence2Provider, FLORENCE_AVAILABLE
                if not FLORENCE_AVAILABLE:
                    self.after(0, lambda: messagebox.showerror(
                        "Not Available",
                        "Florence-2 requires GPU.\n\n"
                        "Install: pip install transformers torch"
                    ))
                    return
                provider = Florence2Provider()

            elif provider_key == 'qwen2vl':
                from ocr.providers.qwen2vl_provider import Qwen2VLProvider, QWEN2VL_AVAILABLE
                if not QWEN2VL_AVAILABLE:
                    self.after(0, lambda: messagebox.showerror(
                        "Not Available",
                        "Qwen2-VL requires GPU.\n\n"
                        "Install: pip install transformers torch qwen-vl-utils"
                    ))
                    return
                provider = Qwen2VLProvider()

            else:
                self.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Unknown provider: {provider_key}"
                ))
                return

            # Check if provider initialized successfully
            if not provider.is_available():
                error = getattr(provider, '_init_error', 'Provider not available')
                self.after(0, lambda e=error: messagebox.showerror(
                    "Not Available",
                    f"{display_name} is not available:\n{e}"
                ))
                return

            # Run OCR
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(provider.process_image(image_path))
            finally:
                loop.close()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Store result in database
            self.db.add_ocr_result(
                image_id=self.current_image_id,
                provider=provider_key,
                text=result.text if result.success else "",
                confidence=result.confidence if result.success else 0.0,
                word_count=result.word_count if result.success else 0,
                char_count=result.char_count if result.success else 0,
                processing_time_ms=elapsed_ms,
                success=result.success,
                error=result.error if not result.success else None
            )

            if result.success:
                cost_info = ""
                if hasattr(result, 'estimated_cost_usd') and result.estimated_cost_usd:
                    cost_info = f", ${result.estimated_cost_usd:.4f}"
                self.after(0, lambda: self._cloud_ocr_finished(
                    True,
                    f"{display_name}: {result.word_count} words in {elapsed_ms:.0f}ms{cost_info}",
                    provider_key
                ))
            else:
                self.after(0, lambda: self._cloud_ocr_finished(
                    False,
                    f"{display_name} failed: {result.error}",
                    provider_key
                ))

        except ImportError as e:
            self.after(0, lambda: messagebox.showerror(
                "Import Error",
                f"Provider not installed: {e}"
            ))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror(
                "Error",
                f"OCR failed: {e}"
            ))
        finally:
            self.after(0, lambda: self.btn_cloud_ocr.config(state=tk.NORMAL))

    def _cloud_ocr_finished(self, success: bool, message: str, provider_key: str):
        """Called when cloud OCR finishes."""
        self.status_bar.config(text=message)

        if success:
            # Refresh provider list to include new provider
            self.providers = self.db.get_providers()
            self.combo_provider['values'] = self.providers

            # Switch to the new provider to show results
            if provider_key in self.providers:
                self.combo_provider.set(provider_key)
                self.current_provider = provider_key
                self._display_ocr_result()

            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Failed", message)

    def on_closing(self):
        """Handle window close."""
        self.tts_stop_flag = True
        if self.db:
            self.db.close()
        self.destroy()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="OCR Results Browser")
    parser.add_argument(
        "--db", "-d",
        default="ocr_results.db",
        help="Path to database file"
    )

    args = parser.parse_args()

    app = OCRBrowser(db_path=args.db)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
