import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve, Eq, sympify, lambdify
from scipy.optimize import linprog
import re

class MejoradoGraficadorInecuaciones:
    def __init__(self, root):
        self.root = root
        self.root.title("Graficador de Inecuaciones Mejorado")
        self.root.geometry("700x700")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.titulos = []
        self.titulos2 = []
        self.titulos3 = []
        self.inecuaciones = []
        self.puntosCorteX = []
        self.puntosCorteY = {}
        self.columnasLocal = 0
        self.valoresX = []
        self.resultados = []
        self.ecuaciones = []
        self.restricciones = []
        self.crear_widgets()
        
    def crear_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Frame para valores de la gráfica
        valores_frame = ttk.LabelFrame(main_frame, text="Valores para la gráfica", padding="10")
        valores_frame.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))
        
        self.startX_var = tk.StringVar(value="-5")  # Valores por defecto
        self.endX_var = tk.StringVar(value="10")
        self.startY_var = tk.StringVar(value="-5")
        self.endY_var = tk.StringVar(value="10")
        
        ttk.Label(valores_frame, text="X Mín:").grid(column=0, row=0, sticky=tk.E, padx=5, pady=5)
        self.startX = tk.Entry(valores_frame, textvariable=self.startX_var, width=10, validate="key")
        self.startX.grid(column=1, row=0, padx=5, pady=5)
        self.startX.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
        
        ttk.Label(valores_frame, text="X Máx:").grid(column=2, row=0, sticky=tk.E, padx=5, pady=5)
        self.endX = tk.Entry(valores_frame, textvariable=self.endX_var, width=10, validate="key")
        self.endX.grid(column=3, row=0, padx=5, pady=5)
        self.endX.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
        
        ttk.Label(valores_frame, text="Y Mín:").grid(column=0, row=1, sticky=tk.E, padx=5, pady=5)
        self.startY = tk.Entry(valores_frame, textvariable=self.startY_var, width=10, validate="key")
        self.startY.grid(column=1, row=1, padx=5, pady=5)
        self.startY.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
        
        ttk.Label(valores_frame, text="Y Máx:").grid(column=2, row=1, sticky=tk.E, padx=5, pady=5)
        self.endY = tk.Entry(valores_frame, textvariable=self.endY_var, width=10, validate="key")
        self.endY.grid(column=3, row=1, padx=5, pady=5)
        self.endY.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
        
        # Frame para inecuaciones
        inecuaciones_frame = ttk.LabelFrame(main_frame, text="Inecuaciones", padding="10")
        inecuaciones_frame.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(inecuaciones_frame, text="Número de inecuaciones:").grid(column=0, row=0, sticky=tk.E, padx=5, pady=5)
        self.num_inecuaciones = ttk.Combobox(inecuaciones_frame, values=list(range(1, 5)), state="readonly", width=5)
        self.num_inecuaciones.grid(column=1, row=0, padx=5, pady=5)
        self.num_inecuaciones.bind("<<ComboboxSelected>>", self.generar_inecuaciones)
        
        self.inecuaciones_container = ttk.Frame(inecuaciones_frame)
        self.inecuaciones_container.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))
        
        # Frame para puntos de corte
        cortes_frame = ttk.LabelFrame(main_frame, text="Puntos de Corte", padding="10")
        cortes_frame.grid(column=1, row=0, rowspan=3, padx=10, pady=10, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        ttk.Label(cortes_frame, text="Número de cortes:").grid(column=0, row=0, sticky=tk.E, padx=5, pady=5)
        self.num_cortes = ttk.Combobox(cortes_frame, values=list(range(2, 11)), state="readonly", width=5)
        self.num_cortes.grid(column=1, row=0, padx=5, pady=5)
        self.num_cortes.bind("<<ComboboxSelected>>", self.generar_puntos_corte)
        
        self.cortes_container = ttk.Frame(cortes_frame)
        self.cortes_container.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))
        
        # Frame para funcion objetivo
        objetivo_frame = ttk.LabelFrame(main_frame, text="Funcion objetivo", padding="10")
        objetivo_frame.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))
        
        self.mostrar_funcion_objetivo = tk.BooleanVar(value=False)
        self.boot = False
        self.boot2 = False
        
        def boleando(event):
            if self.boot:
                if self.maxmin.get() == "Maximizar":
                    self.boot2 = True
                    print(self.boot2)
                elif self.maxmin.get() == "Minimizar":
                    self.boot2 = False
                    print(self.boot2)
                elif self.maxmin.get() == "Selecciona una opción":
                    return messagebox.showinfo("Advertencia", "Seleccione maximizar o minimizar")

        def generar_funcion_objetivo():    
            if self.mostrar_funcion_objetivo.get():
                self.boot = True
                self.check_objetivo.grid(column=0, row=0, columnspan=4, padx=5, pady=5)
                # Crear y guardar las referencias en la instancia
                self.title_objetivo = ttk.Label(objetivo_frame, text="Z = ")
                self.title_objetivo.grid(column=0, row=1, padx=5, pady=5, sticky="w")

                self.coeficiente_x1 = ttk.Entry(objetivo_frame, width=5, validate="key")
                self.coeficiente_x1.grid(column=1, row=1, padx=5, pady=5)
                self.coeficiente_x1.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
                
                self.title_X1 = ttk.Label(objetivo_frame, text="x1 +")
                self.title_X1.grid(column=2, row=1, padx=5, pady=5)

                self.coeficiente_x2 = ttk.Entry(objetivo_frame, width=5, validate="key")
                self.coeficiente_x2.grid(column=3, row=1, padx=5, pady=5)
                self.coeficiente_x2.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
                
                self.title_X2 = ttk.Label(objetivo_frame, text="x2")
                self.title_X2.grid(column=4, row=1, padx=5, pady=5)
                
                values = ["Maximizar", "Minimizar"]
                self.maxmin = ttk.Combobox(objetivo_frame, values=values, state="readonly")
                self.maxmin.grid(column=0, row=2, columnspan=4, padx=5, pady=5)
                self.maxmin.set("Selecciona una opción")
                self.maxmin.bind("<<ComboboxSelected>>", boleando)
                
                self.title_X12 = ttk.Label(objetivo_frame, text="x1")
                self.title_X12.grid(column=0, row=3, padx=5, pady=5)

                self.resultx1 = ttk.Entry(objetivo_frame, width=5, state="readonly")
                self.resultx1.grid(column=1, row=3, padx=5, pady=5)
                
                self.title_X22 = ttk.Label(objetivo_frame, text="x2")
                self.title_X22.grid(column=2, row=3, padx=5, pady=5)

                self.resultx2 = ttk.Entry(objetivo_frame, width=5, state="readonly")
                self.resultx2.grid(column=3, row=3, padx=5, pady=5)
                
                self.title_X32 = ttk.Label(objetivo_frame, text="optima")
                self.title_X32.grid(column=4, row=3, padx=5, pady=5)

                self.resultx3 = ttk.Entry(objetivo_frame, width=10, state="readonly")
                self.resultx3.grid(column=5, row=3, padx=5, pady=5)
                
                self.textResult = ttk.Entry(objetivo_frame, state="readonly")
                self.textResult.grid(column=0, row=4, columnspan=6, padx=5, pady=5, sticky=(tk.W, tk.E))
                
            else:
                self.boot = False
                # Destroy the elements using the instance references
                if hasattr(self, 'title_objetivo'):
                    self.title_objetivo.destroy()
                if hasattr(self, 'coeficiente_x1'):
                    self.coeficiente_x1.destroy()
                if hasattr(self, 'title_X1'):
                    self.title_X1.destroy()
                if hasattr(self, 'coeficiente_x2'):
                    self.coeficiente_x2.destroy()
                if hasattr(self, 'title_X2'):
                    self.title_X2.destroy()
                if hasattr(self, 'title_X12'):
                    self.title_X12.destroy()
                if hasattr(self, 'title_X22'):
                    self.title_X22.destroy()
                if hasattr(self, 'title_X32'):
                    self.title_X32.destroy()
                if hasattr(self, 'resultx1'):
                    self.resultx1.destroy()
                if hasattr(self, 'resultx2'):
                    self.resultx2.destroy()
                if hasattr(self, 'resultx3'):
                    self.resultx3.destroy()
                if hasattr(self, 'maxmin'):
                    self.maxmin.destroy()
                if hasattr(self, 'textResult'):
                    self.textResult.destroy()

        self.check_objetivo = ttk.Checkbutton(objetivo_frame, text="Mostrar función objetivo", variable=self.mostrar_funcion_objetivo, command=generar_funcion_objetivo)
        self.check_objetivo.grid(column=0, row=0, padx=5, pady=5)

        # Frame para restricciones
        restricciones_frame = ttk.LabelFrame(main_frame, text="Restricciones", padding="10")
        restricciones_frame.grid(column=0, row=3, columnspan=2, padx=10, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(restricciones_frame, text="Número de restricciones:").grid(column=0, row=0, sticky=tk.E, padx=5, pady=5)
        self.num_restricciones = ttk.Combobox(restricciones_frame, values=list(range(0, 8)), state="readonly", width=5)
        self.num_restricciones.grid(column=1, row=0, padx=5, pady=5)
        self.num_restricciones.bind("<<ComboboxSelected>>", self.generar_restricciones)
        
        self.restricciones_container = ttk.Frame(restricciones_frame)
        self.restricciones_container.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E))     
        
        # Botón para graficar
        self.boton_graficar = ttk.Button(main_frame, text="Graficar", command=self.graficar_inecuaciones)
        self.boton_graficar.grid(column=0, row=4, columnspan=2, pady=20)
       
    def solo_numeros(self, texto):
        # Permitir si el texto está vacío o contiene solo dígitos o un solo punto decimal
        if texto == "":
            return True
        try:
            float(texto)  # Verificar si el texto es un número válido (entero o decimal)
            return texto.count('.') <= 1  # Permitir solo un punto decimal
        except ValueError:
            return False   
     
    def generar_inecuaciones(self, event):
        for widget in self.inecuaciones_container.winfo_children():
            widget.destroy()
            
        self.inecuaciones.clear()
        num = int(self.num_inecuaciones.get())
        
        for i in range(num):
            frame = ttk.Frame(self.inecuaciones_container)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"Inecuación {i+1}:").pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=30)
            entry.pack(side=tk.LEFT, padx=5)
            self.inecuaciones.append(entry)
    
    def generar_puntos_corte(self, event):
        if len(self.inecuaciones) != 0:
            for widget in self.cortes_container.winfo_children():
                widget.destroy()
            
            self.puntosCorteX.clear()
            self.puntosCorteY.clear()
            
            num_cortes = int(self.num_cortes.get())
            num_inecuaciones = int(self.num_inecuaciones.get())
            
            for i in range(num_cortes):
                frame = ttk.Frame(self.cortes_container)
                frame.pack(fill=tk.X, padx=5, pady=2)
                
                ttk.Label(frame, text=f"X{i+1}:").pack(side=tk.LEFT)
                entry_x = ttk.Entry(frame, width=8, validate="key") 
                entry_x.pack(side=tk.LEFT, padx=5)
                entry_x.configure(validatecommand=(self.root.register(self.solo_numeros), '%P'))
                self.puntosCorteX.append(entry_x)
                
                for j in range(num_inecuaciones):
                    ttk.Label(frame, text=f"Y{j+1}:").pack(side=tk.LEFT)
                    entry_y = ttk.Entry(frame, width=8, state="readonly")
                    entry_y.pack(side=tk.LEFT, padx=5)
                    self.puntosCorteY.setdefault(j+1, []).append(entry_y)
        else:
            messagebox.showinfo("Advertencia", "Seleccione una cantidad de inecuaciones")
    
    def generar_restricciones(self, event):
        for widget in self.restricciones_container.winfo_children():
            widget.destroy()
        
        self.restricciones.clear()
        num = int(self.num_restricciones.get())
        
        for i in range(num):
            frame = ttk.Frame(self.restricciones_container)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"Restricción {i+1}:").pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=30)
            entry.pack(side=tk.LEFT, padx=5)
            self.restricciones.append(entry)

    def graficar_inecuaciones(self):

            # Validar que no estén vacíos
            if not all([self.startX_var.get().strip(), self.endX_var.get().strip(), 
                    self.startY_var.get().strip(), self.endY_var.get().strip()]):
                raise ValueError("Por favor, complete todos los campos") 
            
            # Obtener valores
            ix = self.startX.get().strip()
            fx = self.endX.get().strip()
            iy = self.startY.get().strip()
            fy = self.endY.get().strip()
            
            # Función para extraer el número de np.float64
            def extraer_numero(valor):
                if isinstance(valor, str) and 'np.float64' in valor:
                    # Extraer solo el número entre paréntesis
                    numero = valor.split('(')[1].split(')')[0]
                    return float(numero)
                return float(valor.replace(',', '.'))
            
            # Convertir valores
            x_min = extraer_numero(ix)
            x_max = extraer_numero(fx)
            y_min = extraer_numero(iy)
            y_max = extraer_numero(fy)
            
            print(f"Valores convertidos: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            
            # Validar rangos
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("Los valores mínimos deben ser menores que los máximos")
            
            # Proceder con el cálculo y graficación
            self.calcular_puntos_corte()
            self.graficar(x_min, x_max, y_min, y_max)
            
    def calcular_puntos_corte(self):
        x, y = symbols('x y')
        self.ecuaciones.clear()
        self.resultados.clear()
        
        for entry in self.inecuaciones:
            inecuacion = entry.get()
            ecuacion = self.reemplazar_inecuaciones_por_igualdad(inecuacion)
            despeje_y = solve(ecuacion, y)
            resultado = despeje_y[0]
            self.ecuaciones.append(resultado)
            
            valores_x = [float(entry.get()) for entry in self.puntosCorteX]
            valores_y = [resultado.subs(x, valor_x) for valor_x in valores_x]
            self.resultados.append(valores_y)
        
        for col, valores_y in enumerate(self.resultados, start=1):
            for entry, valor in zip(self.puntosCorteY[col], valores_y):
                entry.config(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, f"{valor:.2f}")
                entry.config(state="readonly")
    
    def reemplazar_inecuaciones_por_igualdad(self, inecuacion):
        x, y = symbols('x y')
        inecuacion_str = str(inecuacion)
        nueva_ecuacion_str = re.sub(r'(>=|<=|>|<)', '=', inecuacion_str)
        lados = nueva_ecuacion_str.split('=')
        lado_izquierdo = eval(lados[0].strip())
        lado_derecho = eval(lados[1].strip())
        return Eq(lado_izquierdo, lado_derecho)
    
    def evaluar_ecuacion(self, x_valor):
        try:
            return float(self.ecuacion.subs('x', float(x_valor)))
        except Exception as e:
            return None
    
    def graficar(self, x_min, x_max, y_min, y_max):
        try:
            # Crear figura y ejes
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Crear grid de puntos
            X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), 
                            np.linspace(y_min, y_max, 100))
            
            # Máscara inicial (todo verdadero)
            mascara_final = np.ones_like(X, dtype=bool)
            
            # Procesar cada inecuación
            x = symbols('x')
            y = symbols('y')
            
            colores = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
            
            for i, (entry, ecuacion) in enumerate(zip(self.inecuaciones, self.ecuaciones)):
                inecuacion = entry.get().strip()
                color = colores[i % len(colores)]
                
                # Crear máscara para esta inecuación
                mascara = self.crear_mascara_inecuacion(inecuacion, ecuacion, X, Y)
                mascara_final &= mascara
                
                # Graficar línea de la inecuación
                x_vals = np.linspace(x_min, x_max, 1000)
                
                # Evaluar ecuación de manera segura
                def evaluar_punto(val):
                    try:
                        return float(ecuacion.subs(x, float(val)))
                    except:
                        return np.nan
                
                y_vals = np.array([evaluar_punto(val) for val in x_vals])
                
                # Filtrar valores no válidos
                valid_mask = ~np.isnan(y_vals)
                x_valid = x_vals[valid_mask]
                y_valid = y_vals[valid_mask]
                
                if len(x_valid) > 0:  # Solo graficar si hay puntos válidos
                    ax.plot(x_valid, y_valid, color=color, label=f'Inecuación {i+1}')
            
            # Aplicar restricciones adicionales
            if hasattr(self, 'restricciones'):
                for restriccion_entry in self.restricciones:
                    restriccion = restriccion_entry.get().strip()
                    if restriccion:
                        # Crear máscara para cada restricción
                        restriccion_mascara = self.crear_mascara_restriccion(restriccion, X, Y)
                        mascara_final &= restriccion_mascara
                        # Dibujar la restricción en la gráfica
                        self.dibujar_restriccion(restriccion, x_min, x_max, y_min, y_max)
            
            # Sombrear región válida
            if np.any(mascara_final):  # Solo sombrear si hay región válida
                ax.contourf(X, Y, mascara_final, levels=[0.5, 1], colors=['gray'], alpha=0.3)
            
            # Dibujar puntos de corte
            if hasattr(self, 'puntosCorteX') and hasattr(self, 'puntosCorteY'):
                for i, (x_entry, y_entries) in enumerate(zip(self.puntosCorteX, 
                        zip(*[self.puntosCorteY[j+1] for j in range(len(self.ecuaciones))]))):
                    try:
                        x_val = float(x_entry.get())
                        for y_entry in y_entries:
                            y_val = float(y_entry.get())
                            ax.plot(x_val, y_val, 'ko', markersize=8)
                            ax.annotate(f'({x_val:.1f}, {y_val:.1f})',
                                        (x_val, y_val),
                                        xytext=(5, 5),
                                        textcoords='offset points')
                    except (ValueError, TypeError):
                        continue
            
            # Si todo sale bien deberia mostrar el punto optimo
            if self.boot2 == True:
                print("Toy tru")
                x1 = self.coeficiente_x1.get()
                x2 = self.coeficiente_x2.get()
                print(f"x1={x1} x2={x2}")
                objetivo = [float(x1), float(x2)]
                inecua = self.inecuaciones
                inecuaciones = [entry.get() for entry in inecua]
                restri = self.restricciones
                restricciones = [entry.get() for entry in restri]
                self.optimizarMaximizar(inecuaciones, restricciones, objetivo)
            elif self.boot2 == False:
                print("Estoy falso ;C")
                x1 = self.coeficiente_x1.get()
                x2 = self.coeficiente_x2.get()
                print(f"x1={x1} x2={x2}")
                objetivo = [float(x1), float(x2)]
                inecua = self.inecuaciones
                inecuaciones = [entry.get() for entry in inecua]
                restri = self.restricciones
                restricciones = [entry.get() for entry in restri]
                self.optimizarMinimizar(inecuaciones, restricciones, objetivo)
            
            # Configurar gráfica
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Mostrar ejes
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Ajustar leyenda
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.show()
            return fig, ax
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar: {str(e)}")
            print(f"Error en graficar: {e}")
            return None, None

        
    def dibujar_restriccion(self, restriccion, x_min, x_max, y_min, y_max):
        try:
            if 'x' in restriccion and '=' in restriccion:
                x_valor = float(restriccion.split('=')[-1].strip())
                if x_min <= x_valor <= x_max:
                    plt.axvline(x=x_valor, color='k', linestyle='--', 
                            label=f'Restricción: {restriccion}')
            elif 'y' in restriccion and '=' in restriccion:
                y_valor = float(restriccion.split('=')[-1].strip())
                if y_min <= y_valor <= y_max:
                    plt.axhline(y=y_valor, color='k', linestyle='--', 
                            label=f'Restricción: {restriccion}')
        except Exception as e:
            print(f"Error al dibujar restricción {restriccion}: {e}")
    
    def crear_mascara_restriccion(self, restriccion, X, Y):
        try:
            # Identificar el tipo de desigualdad
            if '>=' in restriccion:
                signo = '>='
                lado_correcto = 1
            elif '<=' in restriccion:
                signo = '<='
                lado_correcto = -1
            else:
                # Para ecuaciones de igualdad
                return np.abs(self.evaluar_expresion(restriccion, X, Y)) < 1e-10

            # Separar la expresión
            partes = restriccion.split(signo)
            expr_izq = partes[0].strip()
            expr_der = partes[1].strip() if len(partes) > 1 else '0'

            # Convertir a forma estándar: expr_izq - expr_der >= 0 o <= 0
            expr = f"({expr_izq}) - ({expr_der})"
            
            # Crear variables simbólicas y función
            x, y = symbols('x y')
            f = lambdify((x, y), sympify(expr), 'numpy')
            
            # Evaluar y crear máscara
            Z = f(X, Y)
            if lado_correcto == 1:  # Para >=
                return Z >= 0
            else:  # Para <=
                return Z <= 0
            
        except Exception as e:
            print(f"Error en crear_mascara_restriccion: {e}")
            return np.zeros_like(X, dtype=bool)

    def evaluar_expresion(self, expr, X, Y):
        try:
            x, y = symbols('x y')
            f = lambdify((x, y), sympify(expr), 'numpy')
            return f(X, Y)
        except Exception as e:
            print(f"Error en evaluar_expresion: {e}")
            return np.zeros_like(X)
    
    def crear_mascara_inecuacion(self, inecuacion, ecuacion, X, Y):
        try:
            # Convertir la ecuación simbólica a una función numpy
            def evaluar_ecuacion(x_val):
                try:
                    return float(ecuacion.subs('x', float(x_val)))
                except Exception:
                    return np.nan

            evaluar_vectorizado = np.vectorize(evaluar_ecuacion)
            
            # Calcular los valores de y para cada x
            Y_calculado = evaluar_vectorizado(X)
            
            # Crear la máscara según el operador de la inecuación
            if '>=' in inecuacion:
                mascara = Y >= Y_calculado
            elif '<=' in inecuacion:
                mascara = Y <= Y_calculado
            elif '<' in inecuacion:
                mascara = Y < Y_calculado
            elif '>' in inecuacion:
                mascara = Y > Y_calculado
            else:
                mascara = np.ones_like(X, dtype=bool)
            
            # Manejar valores NaN
            mascara = np.where(np.isnan(Y_calculado), False, mascara)
            
            return mascara
        except Exception as e:
            print(f"Error en crear_mascara_inecuacion: {e}")
            return np.ones_like(X, dtype=bool)
    
    def filtrar_restricciones(self, restricciones):
        # Crear una lista para almacenar las restricciones válidas
        restricciones_filtradas = []
        
        # Iterar sobre las restricciones
        for restriccion in restricciones:
            # Eliminar todos los espacios en blanco
            restriccion_sin_espacios = restriccion.replace(" ", "")
            
            # Verificar si la restricción es igual a alguna de las no deseadas
            if restriccion_sin_espacios in ["x>=0", "x<=0", "y>=0", "y<=0"]:
                continue  # Si es una restricción no deseada, se omite
            
            # Agregar la restricción válida a la lista
            restricciones_filtradas.append(restriccion)
        
        return restricciones_filtradas
    
    def extraer_coeficientes(self, inecuaciones):
        A = []  # Matriz de coeficientes
        b = []  # Array para los términos constantes

        for ineq in inecuaciones:
            # Separar la inecuación en parte izquierda (coeficientes) y derecha (constante)
            partes = re.split(r'(<=|>=|<|>)', ineq)
            if len(partes) < 3:
                raise ValueError(f"La inecuación '{ineq}' no tiene un formato válido.")

            coef_str, const_str = partes[0], partes[2]

            # Inicializar coeficientes para x y y
            coef_x = 0.0
            coef_y = 0.0

            # Buscar coeficientes para x e y
            match_x = re.search(r'([+-]?\d*\.?\d*)\s*\*?\s*x', coef_str.replace(" ", ""))
            match_y = re.search(r'([+-]?\d*\.?\d*)\s*\*?\s*y', coef_str.replace(" ", ""))

            # Procesar el coeficiente de x
            if match_x:
                coef_value = match_x.group(1)
                if coef_value in ["", "+"]:
                    coef_x = 1.0
                elif coef_value == "-":
                    coef_x = -1.0
                else:
                    coef_x = float(coef_value)

            # Procesar el coeficiente de y
            if match_y:
                coef_value = match_y.group(1)
                if coef_value in ["", "+"]:
                    coef_y = 1.0
                elif coef_value == "-":
                    coef_y = -1.0
                else:
                    coef_y = float(coef_value)

            # Guardar los coeficientes y la constante
            A.append([coef_x, coef_y])
            b.append(float(const_str.strip()))

        return np.array(A), np.array(b)

    # Función para optimizar la función objetivo dado un conjunto de restricciones
    def optimizarMaximizar(self, inecuaciones, restricciones, funcion_objetivo):
        # Extraer coeficientes de restricciones
        print(inecuaciones)
        print(restricciones)
        convertir = inecuaciones + restricciones
        convertir = self.filtrar_restricciones(convertir)
        A, b = self.extraer_coeficientes(convertir)

        # Configurar los límites de las variables (todas >= 0)
        x1_bounds = (0, None)
        x2_bounds = (0, None)

        # Configurar la función objetivo para linprog
        c = np.array(funcion_objetivo)

        c = -c

        # Resolver el problema de optimización con linprog
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x1_bounds, x2_bounds), method='simplex')

        # Verificar si la solución es ilimitada o acotada
        if res.status == 3:
            print(res)
            print("El problema tiene solución ilimitada (no acotada).")
            if hasattr(self, 'resultx1') and hasattr(self, 'resultx2') and hasattr(self, 'resultx3') and hasattr(self, 'textResult'):
                self.textResult.config(state="normal")
                self.textResult.delete(0, "end")
                self.textResult.insert(0, "El problema tiene solución ilimitada (no acotada).")
                self.textResult.config(state="readonly")
        else:
            print(res)
            # Si la solución es acotada, mostrarla en el gráfico
            vertice_optimo = (res.x[0], res.x[1])
            plt.plot(vertice_optimo[0], vertice_optimo[1], 'ro', markersize=10)
            # Multiplicar res.fun por -1 si se realizó una maximización
            valor_optimo = -res.fun
            print('La solución óptima se alcanza en x1 =', res.x[0], 'y x2 =', res.x[1], 'con un valor de', valor_optimo)
            if hasattr(self, 'resultx1') and hasattr(self, 'resultx2') and hasattr(self, 'resultx3') and hasattr(self, 'textResult'):
                # Limpiar cualquier valor existente y luego insertar el nuevo valor
                self.resultx1.config(state="normal")  # Hacer el campo editable temporalmente
                self.resultx1.delete(0, "end")
                self.resultx1.insert(0, res.x[0])
                self.resultx1.config(state="readonly")  # Volver a hacerlo de solo lectura

                self.resultx2.config(state="normal")
                self.resultx2.delete(0, "end")
                self.resultx2.insert(0, res.x[1])
                self.resultx2.config(state="readonly")
                
                self.resultx3.config(state="normal")
                self.resultx3.delete(0, "end")
                self.resultx3.insert(0, valor_optimo)
                self.resultx3.config(state="readonly")
                
                self.textResult.config(state="normal")
                self.textResult.delete(0, "end")
                self.textResult.insert(0, "Correcta maximizacion")
                self.textResult.config(state="readonly")
            
    def extraer_coeficientes(self, inecuaciones):
        A = []  # Matriz de coeficientes
        b = []  # Array para los términos constantes

        for ineq in inecuaciones:
            # Separar la inecuación en parte izquierda (coeficientes) y derecha (constante)
            partes = re.split(r'(<=|>=|<|>)', ineq)
            if len(partes) < 3:
                raise ValueError(f"La inecuación '{ineq}' no tiene un formato válido.")

            coef_str, const_str = partes[0], partes[2]
            operador = partes[1].strip()  # Captura el operador (<=, >=, <, >)

            # Inicializar coeficientes para x y y
            coef_x = 0.0
            coef_y = 0.0

            # Buscar coeficientes para x e y
            match_x = re.search(r'([+-]?\d*\.?\d*)\s*\*?\s*x', coef_str.replace(" ", ""))
            match_y = re.search(r'([+-]?\d*\.?\d*)\s*\*?\s*y', coef_str.replace(" ", ""))

            # Procesar el coeficiente de x
            if match_x:
                coef_value = match_x.group(1)
                if coef_value in ["", "+"]:
                    coef_x = 1.0
                elif coef_value == "-":
                    coef_x = -1.0
                else:
                    coef_x = float(coef_value)

            # Procesar el coeficiente de y
            if match_y:
                coef_value = match_y.group(1)
                if coef_value in ["", "+"]:
                    coef_y = 1.0
                elif coef_value == "-":
                    coef_y = -1.0
                else:
                    coef_y = float(coef_value)

            # Si el operador es >=, invertir los signos de los coeficientes
            if operador == '>=':
                coef_x = -coef_x
                coef_y = -coef_y
                const_str = str(-float(const_str.strip()))  # Invertir el signo de la constante

            # Guardar los coeficientes y la constante
            A.append([coef_x, coef_y])
            b.append(float(const_str.strip()))

        return np.array(A), np.array(b)
    
    #funcion para minimizar
    def optimizarMinimizar(self, inecuaciones, restricciones, funcion_objetivo):
        # Extraer coeficientes de restricciones
        print(inecuaciones)
        print(restricciones)
        convertir = inecuaciones + restricciones
        convertir = self.filtrar_restricciones(convertir)
        A, b = self.extraer_coeficientes(convertir)
        
        c = np.array(funcion_objetivo)
        
        # Restricciones de las variables x y y (ambas >= 0)
        x0_bounds = (0, None)
        x1_bounds = (0, None)

        # Resolver el problema utilizando linprog
        res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='simplex')
        # Mostrar el resultado
        if res.success:
            if res.status == 3:
                print(res)
                print("El problema tiene solución ilimitada (no acotada).")
                if hasattr(self, 'resultx1') and hasattr(self, 'resultx2') and hasattr(self, 'resultx3') and hasattr(self, 'textResult'):
                    self.textResult.config(state="normal")
                    self.textResult.delete(0, "end")
                    self.textResult.insert(0, "El problema tiene solución ilimitada (no acotada).")
                    self.textResult.config(state="readonly")
            else:
                print(res)
                # Si la solución es acotada, mostrarla en el gráfico
                vertice_optimo = (res.x[0], res.x[1])
                plt.plot(vertice_optimo[0], vertice_optimo[1], 'ro', markersize=10)
                # Multiplicar res.fun por -1 si se realizó una maximización
                valor_optimo = res.fun
                print('La solución óptima se alcanza en x1 =', res.x[0], 'y x2 =', res.x[1], 'con un valor de', valor_optimo)
                if hasattr(self, 'resultx1') and hasattr(self, 'resultx2') and hasattr(self, 'resultx3') and hasattr(self, 'textResult'):
                    # Limpiar cualquier valor existente y luego insertar el nuevo valor
                    self.resultx1.config(state="normal")  # Hacer el campo editable temporalmente
                    self.resultx1.delete(0, "end")
                    self.resultx1.insert(0, res.x[0])
                    self.resultx1.config(state="readonly")  # Volver a hacerlo de solo lectura

                    self.resultx2.config(state="normal")
                    self.resultx2.delete(0, "end")
                    self.resultx2.insert(0, res.x[1])
                    self.resultx2.config(state="readonly")
                    
                    self.resultx3.config(state="normal")
                    self.resultx3.delete(0, "end")
                    self.resultx3.insert(0, valor_optimo)
                    self.resultx3.config(state="readonly")
                    
                    self.textResult.config(state="normal")
                    self.textResult.delete(0, "end")
                    self.textResult.insert(0, "Correcta minimizacion")
                    self.textResult.config(state="readonly")
        else:
            print("No se pudo encontrar una solución óptima.")
    
if __name__ == "__main__":
    root = tk.Tk()
    app = MejoradoGraficadorInecuaciones(root)
    root.mainloop()