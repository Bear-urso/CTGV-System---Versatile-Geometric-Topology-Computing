#!/usr/bin/env python3
"""
CTGV System - Graphical User Interface
Interface visual para configuração e execução de simulações
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

from ctgv import (
    Shape, Gebit, CTGVEngine, GeometricDataModeler,
    TemporalBindingArbiter, ClarificationEngine, visualize_ctgv_processing
)

class CTGV_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CTGV System - Interface Gráfica")
        self.root.geometry("1200x800")

        # Dados do sistema
        self.gebits = []
        self.connections = []
        self.engine = CTGVEngine()
        self.selected_gebit = None

        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        """Configura a interface principal"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Barra de ferramentas
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # Botões de criação
        ttk.Button(toolbar, text="➕ Novo Gebit", command=self.create_gebit_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="🔗 Conectar", command=self.connect_gebits_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="▶️ Executar", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="🗑️ Limpar", command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Painel esquerdo - Lista de Gebits
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left_panel, text="Gebits Criados:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.gebit_listbox = tk.Listbox(left_panel, width=30, height=20)
        self.gebit_listbox.pack(fill=tk.BOTH, expand=True)
        self.gebit_listbox.bind('<<ListboxSelect>>', self.on_gebit_select)

        # Botão remover
        ttk.Button(left_panel, text="Remover Selecionado", command=self.remove_gebit).pack(fill=tk.X, pady=(5, 0))

        # Painel direito - Visualização e controles
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Notebook para abas
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Aba de Rede
        network_frame = ttk.Frame(self.notebook)
        self.notebook.add(network_frame, text="Rede")

        # Canvas para visualização da rede
        self.network_canvas = tk.Canvas(network_frame, bg='white', width=600, height=400)
        self.network_canvas.pack(fill=tk.BOTH, expand=True)

        # Aba de Parâmetros
        params_frame = ttk.Frame(self.notebook)
        self.notebook.add(params_frame, text="Parâmetros")

        self.setup_params_tab(params_frame)

        # Aba de Clarificação
        clarification_frame = ttk.Frame(self.notebook)
        self.notebook.add(clarification_frame, text="Clarificação")

        self.setup_clarification_tab(clarification_frame)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_clarification_tab(self, parent):
        """Configura a aba de clarificação"""
        # Clarificação de Decisões
        decision_frame = ttk.LabelFrame(parent, text="Clarificação de Decisões")
        decision_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(decision_frame, text="Usar Clarificação:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.use_clarification_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(decision_frame, variable=self.use_clarification_var).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(decision_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.clarification_threshold_var = tk.DoubleVar(value=0.7)
        ttk.Entry(decision_frame, textvariable=self.clarification_threshold_var).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(decision_frame, text="Max Rounds:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_clarification_rounds_var = tk.IntVar(value=5)
        ttk.Entry(decision_frame, textvariable=self.max_clarification_rounds_var).grid(row=2, column=1, padx=5, pady=2)

        # Clarificação de Padrões
        pattern_frame = ttk.LabelFrame(parent, text="Clarificação de Padrões")
        pattern_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(pattern_frame, text="Foco:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.clarification_focus_var = tk.StringVar(value="symmetry")
        focus_combo = ttk.Combobox(pattern_frame, textvariable=self.clarification_focus_var,
                                  values=["symmetry", "structure", "noise"])
        focus_combo.grid(row=0, column=1, padx=5, pady=2)

        ttk.Button(pattern_frame, text="Aplicar Clarificação ao Padrão",
                  command=self.apply_pattern_clarification).grid(row=1, column=0, columnspan=2, pady=10)

    def create_gebit_dialog(self):
        """Diálogo para criar novo Gebit"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Criar Novo Gebit")
        dialog.geometry("300x200")

        ttk.Label(dialog, text="Forma:").pack(pady=5)
        shape_var = tk.StringVar()
        shape_combo = ttk.Combobox(dialog, textvariable=shape_var,
                                  values=[s.name for s in Shape])
        shape_combo.pack()
        shape_combo.set("ORIGIN")

        ttk.Label(dialog, text="Intensidade:").pack(pady=5)
        intensity_var = tk.DoubleVar(value=1.0)
        ttk.Entry(dialog, textvariable=intensity_var).pack()

        ttk.Label(dialog, text="Rótulo:").pack(pady=5)
        label_var = tk.StringVar(value="")
        ttk.Entry(dialog, textvariable=label_var).pack()

        def create():
            try:
                shape = Shape[shape_var.get()]
                gebit = Gebit(shape, intensity_var.get(), label_var.get())
                self.gebits.append(gebit)
                self.update_display()
                dialog.destroy()
                self.status_var.set(f"Gebit {gebit.label} criado")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao criar Gebit: {e}")

        ttk.Button(dialog, text="Criar", command=create).pack(pady=10)

    def connect_gebits_dialog(self):
        """Diálogo para conectar Gebits"""
        if len(self.gebits) < 2:
            messagebox.showwarning("Aviso", "Precisa de pelo menos 2 Gebits para conectar")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Conectar Gebits")
        dialog.geometry("300x150")

        ttk.Label(dialog, text="Gebit Origem:").pack(pady=5)
        from_var = tk.StringVar()
        from_combo = ttk.Combobox(dialog, textvariable=from_var,
                                 values=[g.label for g in self.gebits])
        from_combo.pack()

        ttk.Label(dialog, text="Gebit Destino:").pack(pady=5)
        to_var = tk.StringVar()
        to_combo = ttk.Combobox(dialog, textvariable=to_var,
                               values=[g.label for g in self.gebits])
        to_combo.pack()

        ttk.Label(dialog, text="Peso:").pack(pady=5)
        weight_var = tk.DoubleVar(value=1.0)
        ttk.Entry(dialog, textvariable=weight_var).pack()

        def connect():
            try:
                from_gebit = next(g for g in self.gebits if g.label == from_var.get())
                to_gebit = next(g for g in self.gebits if g.label == to_var.get())

                if from_gebit.connect_to(to_gebit, weight_var.get()):
                    self.connections.append((from_gebit, to_gebit, weight_var.get()))
                    self.update_display()
                    dialog.destroy()
                    self.status_var.set(f"Conexão criada: {from_var.get()} → {to_var.get()}")
                else:
                    messagebox.showerror("Erro", "Não foi possível criar conexão (restrições geométricas)")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao conectar: {e}")

        ttk.Button(dialog, text="Conectar", command=connect).pack(pady=10)

    def remove_gebit(self):
        """Remove o Gebit selecionado"""
        selection = self.gebit_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        gebit = self.gebits[index]

        # Remove conexões relacionadas
        self.connections = [(f, t, w) for f, t, w in self.connections
                           if f != gebit and t != gebit]

        # Remove o Gebit
        del self.gebits[index]
        self.update_display()
        self.status_var.set(f"Gebit {gebit.label} removido")

    def on_gebit_select(self, event):
        """Quando um Gebit é selecionado na lista"""
        selection = self.gebit_listbox.curselection()
        if selection:
            self.selected_gebit = self.gebits[selection[0]]
            self.status_var.set(f"Selecionado: {self.selected_gebit.label}")

    def update_display(self):
        """Atualiza todas as exibições"""
        # Atualiza lista de Gebits
        self.gebit_listbox.delete(0, tk.END)
        for gebit in self.gebits:
            self.gebit_listbox.insert(tk.END, f"{gebit.label} ({gebit.shape.name})")

        # Atualiza visualização da rede
        self.draw_network()

    def draw_network(self):
        """Desenha a rede de Gebits no canvas"""
        self.network_canvas.delete("all")

        if not self.gebits:
            return

        # Posicionamento simples em círculo
        center_x, center_y = 300, 200
        radius = 150
        angle_step = 2 * 3.14159 / len(self.gebits) if self.gebits else 0

        positions = {}
        for i, gebit in enumerate(self.gebits):
            angle = i * angle_step
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions[gebit] = (x, y)

            # Desenha o Gebit
            color = self.get_shape_color(gebit.shape)
            self.network_canvas.create_oval(x-20, y-20, x+20, y+20, fill=color, outline='black')
            self.network_canvas.create_text(x, y, text=gebit.label, font=("Arial", 8))

        # Desenha conexões
        for from_gebit, to_gebit, weight in self.connections:
            if from_gebit in positions and to_gebit in positions:
                x1, y1 = positions[from_gebit]
                x2, y2 = positions[to_gebit]
                self.network_canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2)
                # Peso da conexão
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                self.network_canvas.create_text(mid_x, mid_y, text=f"{weight:.1f}",
                                              font=("Arial", 7), fill='blue')

    def get_shape_color(self, shape):
        """Retorna cor para cada forma"""
        colors = {
            Shape.ORIGIN: 'red',
            Shape.FLOW: 'blue',
            Shape.DECISOR: 'green',
            Shape.MEMORY: 'yellow',
            Shape.RESONATOR: 'purple',
            Shape.AMPLIFIER: 'orange',
            Shape.INHIBITOR: 'black',
            Shape.TRANSFORMER: 'cyan',
            Shape.LOOP: 'magenta',
            Shape.SENSOR: 'brown'
        }
        return colors.get(shape, 'gray')

    def run_simulation(self):
        """Executa a simulação"""
        if not self.gebits:
            messagebox.showwarning("Aviso", "Crie pelo menos um Gebit primeiro")
            return

        # Atualiza parâmetros do engine
        self.engine = CTGVEngine(
            threshold=self.threshold_var.get(),
            max_iterations=self.max_iter_var.get(),
            use_superposition=self.superposition_var.get()
        )

        # Executa em thread separada para não travar a UI
        def run():
            try:
                self.status_var.set("Executando simulação...")
                result = self.engine.propagate(self.gebits)

                # Aplicar clarificação se ativada
                if self.use_clarification_var.get():
                    self.status_var.set("Aplicando clarificação...")
                    clarifier = ClarificationEngine(self.engine, self.clarification_threshold_var.get())
                    clarification_result = clarifier.clarify_decision(
                        self.gebits,
                        max_clarification_rounds=self.max_clarification_rounds_var.get()
                    )

                    # Atualizar resultados com clarificação
                    result['clarification'] = clarification_result

                # Mostra resultados
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Simulação Concluída\n\n")
                self.results_text.insert(tk.END, f"Iterações: {result['iterations']}\n")
                self.results_text.insert(tk.END, f"Convergência: {result['converged']}\n")
                self.results_text.insert(tk.END, f"Coerência Global: {result['global_coherence']:.4f}\n\n")
                self.results_text.insert(tk.END, "Estados Finais:\n")

                for label, state in result['final_states'].items():
                    if state > 0.001:  # Só mostra estados significativos
                        self.results_text.insert(tk.END, f"  {label}: {state:.4f}\n")

                # Mostra resultados de clarificação
                if 'clarification' in result:
                    clarification = result['clarification']
                    self.results_text.insert(tk.END, f"\nClarificação Aplicada:\n")
                    self.results_text.insert(tk.END, f"  Rounds: {clarification['clarification_rounds']}\n")
                    self.results_text.insert(tk.END, f"  Confidence: {clarification['confidence']:.4f}\n")
                    self.results_text.insert(tk.END, f"  Ambiguity Reduction: {clarification['ambiguity_reduction']:.4f}\n")

                    if clarification['clarified_decision']:
                        self.results_text.insert(tk.END, f"  Decisão Final: {clarification['clarified_decision'].label}\n")

                self.status_var.set("Simulação concluída")
                self.update_display()  # Atualiza visualização

            except Exception as e:
                messagebox.showerror("Erro", f"Erro na simulação: {e}")
                self.status_var.set("Erro na simulação")

        threading.Thread(target=run, daemon=True).start()

    def apply_pattern_clarification(self):
        """Aplica clarificação ao padrão inserido"""
        try:
            pattern_text = self.pattern_text.get(1.0, tk.END).strip()
            if not pattern_text:
                messagebox.showwarning("Aviso", "Insira um padrão primeiro")
                return

            pattern = eval(pattern_text)
            pattern = np.array(pattern, dtype=np.float32)

            # Aplicar clarificação
            clarifier = ClarificationEngine(self.engine, self.clarification_threshold_var.get())
            clarification_result = clarifier.clarify_pattern(
                pattern, self.clarification_focus_var.get()
            )

            # Mostra resultados
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Clarificação de Padrão Aplicada\n\n")
            self.results_text.insert(tk.END, f"Foco: {clarification_result['clarification_focus']}\n")
            self.results_text.insert(tk.END, f"Coerência Inicial: {clarification_result['initial_coherence']:.4f}\n")
            self.results_text.insert(tk.END, f"Coerência Final: {clarification_result['final_coherence']:.4f}\n")
            self.results_text.insert(tk.END, f"Melhoria: {clarification_result['coherence_improvement']:.4f}\n")
            self.results_text.insert(tk.END, f"Clareza do Padrão: {clarification_result['pattern_clarity']:.4f}\n\n")

            self.results_text.insert(tk.END, f"Padrão Original:\n{clarification_result['original_pattern']}\n\n")
            self.results_text.insert(tk.END, f"Padrão Clarificado:\n{clarification_result['clarified_pattern']}")

            self.status_var.set("Clarificação aplicada ao padrão")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro na clarificação: {e}")

    def clear_all(self):
        """Limpa todos os Gebits e conexões"""
        self.gebits.clear()
        self.connections.clear()
        self.results_text.delete(1.0, tk.END)
        self.update_display()
        self.status_var.set("Sistema limpo")

def main():
    root = tk.Tk()
    app = CTGV_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()