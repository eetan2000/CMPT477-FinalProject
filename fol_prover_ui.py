import tkinter as tk
from tkinter import ttk, messagebox

# Import logic here

from fol_logic import prove_formula


class FOLProverUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FOL Resolution Prover")
        self.root.geometry("900x600")

        # ---- Top panel: controls ----
        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(side=tk.TOP, fill=tk.X)

        # Formula label
        label = ttk.Label(input_frame, text="Input Formula:")
        label.grid(row=0, column=0, sticky="w")

        # Max steps
        max_steps_label = ttk.Label(input_frame, text="Max steps:")
        max_steps_label.grid(row=0, column=1, padx=(20, 0), sticky="e")

        self.max_steps_var = tk.StringVar(value="500")
        max_steps_entry = ttk.Entry(input_frame, width=8, textvariable=self.max_steps_var)
        max_steps_entry.grid(row=0, column=2, sticky="w")

        # Buttons
        prove_button = ttk.Button(input_frame, text="Prove Validity", command=self.on_prove_clicked)
        prove_button.grid(row=0, column=3, padx=(20, 0))

        clear_button = ttk.Button(input_frame, text="Clear", command=self.on_clear_clicked)
        clear_button.grid(row=0, column=4, padx=(10, 0))

        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(5, weight=1)

        # ---- Formula text area ----
        formula_frame = ttk.LabelFrame(root, text="Formula", padding=10)
        formula_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        self.formula_text = tk.Text(formula_frame, height=6, wrap=tk.WORD)
        self.formula_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        formula_scroll = ttk.Scrollbar(formula_frame, orient="vertical", command=self.formula_text.yview)
        formula_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.formula_text.configure(yscrollcommand=formula_scroll.set)

        # ---- Output text area ----
        output_frame = ttk.LabelFrame(root, text="Proof Output", padding=10)
        output_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, state="normal")
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=output_scroll.set)

        # ---- Status bar ----
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)



    def on_prove_clicked(self):
        formula = self.formula_text.get("1.0", tk.END)
        max_steps_str = self.max_steps_var.get().strip()

        try:
            max_steps = int(max_steps_str)
            if max_steps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Max steps must be a positive integer.")
            return

        if not formula.strip():
            messagebox.showwarning("Warning", "Please input a formula first.")
            return

        self.status_var.set("Proving...")
        self.root.update_idletasks()

        # ---- Call logic here ----
        try:
            result = prove_formula(formula, max_steps=max_steps)
        except Exception as e:
            # Catch any unexpected errors from your logic
            self._show_error(f"Exception in prover: {e}")
            self.status_var.set("Error.")
            return

        # Normalize result
        is_valid = result.get("is_valid")
        message = result.get("message", "")
        steps = result.get("steps", [])

        # ---- Show in UI ----
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)

        # Header line based on is_valid
        if is_valid is True:
            summary = "RESULT: VALID (refutation of ¬F found).\n"
        elif is_valid is False:
            summary = "RESULT: NOT PROVED VALID (satisfiable or unknown).\n"
        else:
            summary = "RESULT: ERROR or UNKNOWN.\n"

        self.output_text.insert(tk.END, summary)
        self.output_text.insert(tk.END, f"Message: {message}\n\n")

        if steps:
            self.output_text.insert(tk.END, "=== Proof steps ===\n")
            for step in steps:
                self.output_text.insert(tk.END, step + "\n")
        else:
            self.output_text.insert(tk.END, "(No steps returned.)\n")

        self.output_text.see(tk.END)
        self.status_var.set("Done.")

    def on_clear_clicked(self):
        self.formula_text.delete("1.0", tk.END)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.status_var.set("Cleared.")

    def _show_error(self, msg: str):
        messagebox.showerror("Error", msg)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "RESULT: ERROR.\n")
        self.output_text.insert(tk.END, f"Message: {msg}\n")


def main():
    root = tk.Tk()
    app = FOLProverUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
