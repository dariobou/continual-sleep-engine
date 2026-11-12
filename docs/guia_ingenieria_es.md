# Especificación Maestra y Guía de Ingeniería: Continual Sleep Engine
**Subtítulo:** *Edge-to-HPC Asynchronous Memory Consolidation for SLMs with Automated Alignment & Safety Quality Gates*  
**Autor:** Estudiante de Ingeniería Informática — Especialidad en Ingeniería del Software (UPV)  
**Objetivo Profesional:** Proyecto insignia (*Flagship Portfolio*) para roles de **AI Engineer** (Sistemas/MLOps/LLMOps) y **AI Safety Engineer** (Alineación, Evaluación y Robustez).

---

## 0. Nota Preliminar: Tu Ventaja Competitiva como Ingeniero del Software

Antes de tocar una sola línea de código o hablar de tensores, conviene aclarar un mito de la industria actual: **tener una base de Ingeniería del Software no es una debilidad para entrar en IA; es tu mayor ventaja competitiva si sabes conectarla.**

El ecosistema de IA está lleno de *notebooks* de Jupyter caóticos que funcionan una vez y se rompen al pasar a producción. Lo que las empresas y laboratorios de AI Safety buscan desesperadamente son ingenieros capaces de tratar a los modelos de lenguaje como **componentes de software sujetos a contratos de datos, pruebas de regresión, control de versiones, trazabilidad y cortafuegos de seguridad (*Quality Gates*)**.

### Piedra Rosetta: De Ingeniería del Software a Ingeniería de IA

Para que ningún concepto técnico te resulte ajeno, a lo largo de todo el proyecto utilizaremos esta equivalencia directa con conceptos que ya dominas en la carrera:

| Concepto en Ingeniería del Software | Equivalente en *Continual Sleep Engine* (IA Moderna) | Por qué importa en este proyecto |
| :--- | :--- | :--- |
| **Código fuente (`main` branch)** | **Modelo Base en FP16/BF16 (`Safetensors`)** | Es la "fuente de la verdad" en alta precisión que vive en el servidor/cluster. Nunca se destruye. |
| **Parche / Feature Branch (`git diff`)** | **Adaptador LoRA ($\Delta W = B \times A$)** | En vez de modificar los 3.000 millones de parámetros, entrenamos un "parche" diminuto ($<1\%$ del tamaño) con lo aprendido en el día. |
| **`git merge` a la rama principal** | **Weight Merging (`merge_and_unload`)** | Operación algebraica ($W_{\text{nuevo}} = W_{\text{base}} + B \times A$) que integra el parche en el modelo sin añadir latencia en tiempo de ejecución. |
| **Compilación a binario (`gcc -O3`)** | **Cuantización a formato `.gguf` (`Q4_K_M`)** | Comprime los pesos de 16 bits a 4 bits para que quepan y corran rápido en los 6 GB de tu RTX 3060. **Regla de oro:** *igual que no editas un `.exe` compilado, nunca re-entrenamos directamente sobre un `.gguf`.* |
| **Caché en memoria vs. Base de Datos** | **Ventana de Contexto / RAG (Vigilia) vs. Pesos (Sueño)** | Durante el día el modelo recuerda al instante leyendo un buffer temporal (caché); por la noche consolida el conocimiento en sus pesos (persistencia neuronal). |
| **Pipeline CI/CD y Tests de Regresión** | **Suite de Evaluación de *Forgetting* y *Safety Drift*** | Antes de desplegar el nuevo `.gguf` a tu portátil, pasa por una batería de tests automáticos. Si el modelo se ha vuelto inseguro o ha olvidado cómo razonar, **se aborta el despliegue (*Rollback*)**. |
| **Patrón *Ports & Adapters* (Hexagonal)** | **Ejecución Híbrida (`Local Dry-Run` vs. `SLURM HPC`)** | Desacoplamos la lógica del hardware: podrás probar todo el ciclo en miniatura en tu propio portátil antes de lanzarlo al cluster de la UPV. |

---

## 1. Visión General del Proyecto y Fundamento Científico

### 1.1. El Problema Real que Resolvemos
Los Modelos de Lenguaje Pequeños (*Small Language Models*, SLMs de 1B a 3B de parámetros) ejecutados en hardware local sufren tres limitaciones fundamentales cuando queremos que actúen como asistentes personales persistentes:

1. **Coste cuadrático y saturación de contexto:** Inyectar todo el historial de semanas de trabajo en el *prompt* satura la VRAM (por el crecimiento de la *KV Cache*), aumenta la latencia y degrada la atención del modelo (*Lost in the Middle*).
2. **Olvido Catastrófico (*Catastrophic Forgetting*):** Si intentamos entrenar el modelo cada noche solo con los datos nuevos del día, los gradientes nuevos sobrescriben los pesos previos y el modelo olvida cómo razonar, programar o hablar con fluidez.
3. **Degradación de Alineación (*Safety Drift*):** Demostrado en la literatura reciente de AI Safety (*Qi et al., 2023*), ajustar un modelo mediante *fine-tuning* —incluso con datos 100% benignos e inocentes— altera sutilmente las direcciones de activación encargadas de rechazar peticiones peligrosas, destruyendo el entrenamiento de seguridad original (RLHF/DPO).

### 1.2. ¿Por qué no usar simplemente RAG? (La respuesta para tus entrevistas)
En cualquier entrevista técnica te preguntarán: *"¿Por qué consolidar en los pesos en lugar de usar una base de datos vectorial (RAG)?"*  
Nuestra arquitectura se inspira en la **Teoría de Sistemas de Aprendizaje Complementarios (*Complementary Learning Systems - CLS*)** de la neurociencia cognitiva y en los trabajos de consolidación tipo **SLEEP**:

* **El Hipocampo (Fase de Vigilia — No paramétrico):** Durante el día necesitas memoria episódica instantánea. Si le dices al asistente a las 10:00 *"Mi clave del proyecto X es Alfa"*, a las 10:05 debe saberlo sin esperar a la noche. Por eso durante la vigilia usamos un buffer ligero de sesión en disco (`wake_buffer.jsonl`) que se inyecta dinámicamente.
* **El Neocórtex (Fase de Sueño — Paramétrico):** Si acumulas miles de notas durante meses, el RAG se vuelve ruidoso, lento y no adapta el *comportamiento, estilo ni intuición* del modelo. Durante el sueño, un proceso asíncrono transforma las experiencias del día en múltiples variaciones sintéticas (*Replay*), las mezcla con ejemplos de control y de seguridad, y las destila en los pesos del modelo mediante **LoRA**, vaciando el buffer episódico para el nuevo día.

### 1.3. Arquitectura del Sistema Corregida (Edge-to-HPC con *Safety Gate*)

> **Corrección arquitectónica clave respecto al borrador inicial:**  
> El fichero `.gguf` cuantizado a 4 bits que corre en tu portátil es un **artefacto final de solo lectura** (con pérdida de precisión). El entrenamiento nocturno en el cluster **siempre** parte del checkpoint en alta precisión (`FP16`/`BF16`) almacenado en el cluster, nunca des-cuantizando el `.gguf` del portátil.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. CLIENTE EDGE — FASE DE VIGILIA (Portátil / RTX 3060 6GB VRAM)        │
│    • Ejecuta inferencia rápida con el artefacto actual: model_v{N}.gguf │
│    • Memoria a corto plazo: lee/escribe eventos en wake_buffer.jsonl    │
│    • Filtra datos inválidos o sensibles mediante contratos Pydantic     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                   Fin de jornada: rsync / SSH
                   (Sube solo wake_buffer.jsonl — unos pocos KB)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. NODO HPC UPV (tensor.dsic.upv.es — 4x NVIDIA L40S 48GB / SLURM)      │
│    Cola: -p docencia | Recursos: --gres=shard:2 (~16GB VRAM) | cu126    │
│                                                                         │
│  [A] Generador de Sueños (Synthetic Rehearsal):                         │
│      • Toma hechos crudos de wake_buffer.jsonl                          │
│      • Genera variaciones Q&A multi-perspectiva + razonamiento          │
│                                                                         │
│  [B] Mezclador Anti-Olvido y Anti-Drift (Dataset Mixer):                │
│      • 60% Memorias Sintéticas del Día (Conocimiento nuevo)             │
│      • 30% Replay General + Memorias Pasadas (Evita olvido catastrófico)│
│      • 10% Safety Anchors (Prompts dañinos -> Rechazo seguro)           │
│                                                                         │
│  [C] Consolidación LoRA y Fusión (Sobre pesos FP16/BF16 en el Cluster): │
│      • Entrena adaptador LoRA (r=16, alpha=32)                          │
│      • Fusión matemática en FP16: W_candidato = W_base + B × A          │
│                                                                         │
│  [D] CORTAFUEGOS DE CALIDAD Y SEGURIDAD (CI/CD Quality Gate):           │
│      • Evalúa Retención + Capacidades Generales + Safety Drift          │
│      • ¿Supera todos los umbrales?                                      │
│         ├─ NO ──► ABORTA Y GENERA ALERTA (Se mantiene model_v{N}.gguf)  │
│         └─ SÍ ──► Compila/Cuantiza a Q4_K_M: model_v{N+1}.gguf          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                   rsync / descarga verificada por SHA256
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. DESPERTAR (Wake Up)                                                  │
│    • El portátil actualiza el puntero simbólico a model_v{N+1}.gguf     │
│    • Archiva el wake_buffer.jsonl consolidado y arranca limpio          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Guía de Supervivencia Técnica (Todo lo que necesitas saber explicado desde cero)

Esta sección es tu manual de referencia rápida. No necesitas memorizarlo hoy; volveremos a cada apartado cuando lleguemos a la fase correspondiente.

### 2.1. Aritmética de VRAM: ¿Por qué elegimos un modelo 3B y cuantización Q4?
Un modelo de lenguaje es, en esencia, una colección de matrices llenas de números en coma flotante (llamadas **tensores**).  
Si elegimos `Qwen-2.5-3B-Instruct` o `Llama-3.2-3B-Instruct`, tenemos aproximadamente $3 \times 10^9$ parámetros (3.000 millones de números).

¿Cuánta memoria de tarjeta gráfica (VRAM) ocupa simplemente cargar esos números?
* **En precisión completa (`FP32` — 32 bits = 4 bytes por parámetro):**  
  $3 \times 10^9 \times 4\text{ bytes} \approx 12\text{ GB de VRAM}$. *(Imposible en tu RTX 3060 de 6 GB).*
* **En media precisión (`FP16` o `BF16` — 16 bits = 2 bytes por parámetro):**  
  $3 \times 10^9 \times 2\text{ bytes} \approx 6\text{ GB de VRAM}$. *(Entraría al límite absoluto, pero al primer mensaje se colapsaría con un error `CUDA Out Of Memory` porque no queda espacio para el contexto ni para el sistema operativo).*
* **Cuantizado en 4 bits (`GGUF Q4_K_M` — ~4.5 bits de media por parámetro):**  
  $3 \times 10^9 \times 0.56\text{ bytes} \approx 1.7\text{ GB de VRAM}$.  
  Si le sumamos ~0.8 GB para la memoria de conversación activa (*KV Cache*) y los buffers de CUDA, el consumo total es de **~2.5 GB de VRAM**, dejando más de 3 GB libres en tu portátil y generando texto a más de 40-60 tokens por segundo.

### 2.2. ¿Qué es LoRA y por qué no entrenamos todos los pesos?
Si en la fase de sueño intentáramos modificar directamente los 3.000 millones de parámetros (*Full Fine-Tuning*):
1. Necesitaríamos guardar en VRAM los pesos, los gradientes y dos estados del optimizador `AdamW` por cada parámetro ($\approx 24 - 36\text{ GB de VRAM}$).
2. Con tan pocos datos diarios (ej. 50-100 ejemplos), destruiríamos el conocimiento previo del modelo en segundos (*overfitting* masivo).

**La solución — LoRA (*Low-Rank Adaptation*):**  
Congelamos la matriz original de pesos $W_0$ (de tamaño, por ejemplo, $4096 \times 4096$, que tiene $16.7$ millones de números) para que no se pueda modificar. En paralelo, añadimos dos matrices pequeñitas $A$ (de $16 \times 4096$) y $B$ (de $4096 \times 16$).  
* El número $r = 16$ se llama **rango (*rank*)**.
* Entre $A$ y $B$ solo suman $2 \times 16 \times 4096 = 131.072$ parámetros (¡menos del $0.8\%$ de la matriz original!).
* Durante el sueño solo entrenamos $A$ y $B$. Al terminar la noche, como el producto $B \times A$ tiene exactamente la misma dimensión ($4096 \times 4096$) que $W_0$, simplemente las sumamos:
  $$W_{\text{nuevo}} = W_0 + \frac{\alpha}{r} (B \times A)$$
* **Resultado mágico:** el modelo resultante tiene exactamente la misma arquitectura y velocidad que el original, sin ningún coste extra en inferencia.

### 2.3. ¿Por qué entrenamos con un "Profesor" (*Synthetic Rehearsal*)?
Si durante el día anotas un hecho único en el buffer:  
> *"El servidor de producción del proyecto Aurora usa el puerto 8443 y requiere certificado mTLS."*

Y entrenas una red neuronal pasando esa única frase 10 veces, el modelo solo aprenderá a repetir la frase como un loro si la empiezas igual, pero si le preguntas *"¿Qué necesito para conectarme a Aurora?"*, no sabrá responder.  
Durante la fase de sueño, un modelo generador (o el propio modelo con un *prompt* especializado de extracción) actúa como **Profesor**: lee tu nota cruda y genera entre 5 y 10 pares de Pregunta-Respuesta variados (directos, inversos, de resolución de problemas y de razonamiento) para que el aprendizaje sea semántico y no memorización literal.

### 2.4. Especificaciones Reales del Servidor UPV (`tensor.dsic.upv.es`) y SLURM
A partir de la documentación oficial del DSIC, contamos con una máquina excelente y muy moderna para este proyecto:
* **Hardware del Nodo (`tensor.dsic.upv.es`):** Nodo único con **503 GB de RAM**, hasta **96 CPUs** y **4 GPUs NVIDIA L40S de 48 GB GDDR6 ECC cada una** (arquitectura Ada Lovelace de centro de datos con soporte nativo para `bfloat16`, `FlashAttention-2` y **CUDA 12.6** / `cu126`).
* **Colas de SLURM (`-p`):**
  - `docencia`: Trabajos de menos de 24 horas (nuestra cola principal: un ciclo de sueño completo tardará apenas **5–10 minutos** en una L40S).
  - `long`: Trabajos de más de 24 horas (no la necesitaremos salvo para dejar corriendo de golpe el experimento completo de 5 noches si fuera necesario).
* **El concepto clave de GPUs Lógicas (`--gres=shard:N`):**
  En `tensor.dsic.upv.es`, las 4 GPUs físicas de 48 GB (192 GB de VRAM en total) se dividen en **24 GPUs lógicas (`shards`)**, lo que equivale a **6 shards por cada L40S ($\approx 8\text{ GB de VRAM por shard}$)**:
  - Si pides una GPU física entera (`--gres=gpu:1`), tu trabajo esperará en cola hasta que haya una L40S completamente vacía.
  - En cambio, la recomendación oficial del DSIC es pedir **shards**. Solicitando **`#SBATCH --gres=shard:2`** ($\approx 16\text{ GB}$ de VRAM) o **`#SBATCH --gres=shard:3`** ($\approx 24\text{ GB}$ de VRAM), nuestros trabajos entrarán en ejecución casi al instante sin bloquear a otros compañeros y tendrán memoria suficiente para entrenar el modelo 3B en `bfloat16` y fusionar los pesos.

### 2.5. Reglas de Oro para tu Entorno (`WSL2` $\leftrightarrow$ `tensor.dsic.upv.es`)
1. **Alias SSH para evitar el problema del doble `@` en `rsync`:**  
   El acceso al servidor tiene el formato `ssh login@alumno.upv.es@tensor.dsic.upv.es`. Como el nombre de usuario contiene un `@`, comandos como `rsync` o `scp` pueden fallar al interpretar dos `@`. Lo solucionaremos creando en tu WSL2 el archivo `~/.ssh/config`:
   ```ssh-config
   Host tensor-upv
       HostName tensor.dsic.upv.es
       User TU_LOGIN@alumno.upv.es
   ```
   Con esto, tanto tú como nuestros scripts conectarán simplemente usando `ssh tensor-upv` y `rsync ... tensor-upv:~/continual-sleep-engine/`.
2. **Cuidado con las carpetas montadas `$HOME/W` y `$HOME/asigDSIC`:**  
   En `tensor.dsic.upv.es`, tu `$HOME` contiene dos puntos de montaje de red (`W` y `asigDSIC`). **Nunca** ejecutaremos comandos recursivos sobre `$HOME` sin excluirlas. Para comprobar cuánto espacio ocupamos en el disco compartido (`/home` tiene 14 TB compartidos entre todos los alumnos), usaremos siempre el comando oficial:
   ```bash
   du -sh --exclude=asigDSIC --exclude=W $HOME
   ```
   Además, al terminar cada ciclo de sueño eliminaremos los checkpoints temporales intermedios y conservaremos solo el modelo base en caché (`Qwen2.5-3B-Instruct` ocupa ~6 GB) y el `.gguf` final (~1.9 GB), manteniendo una huella de disco mínima (<12 GB en total).
3. **Plantilla SLURM Oficial para nuestro Proyecto (`cluster/run_sleep.slurm`):**
   ```bash
   #!/bin/bash
   #SBATCH -p docencia                  # Cola para trabajos < 24h
   #SBATCH --gres=shard:2               # 2 GPUs lógicas (~16GB VRAM en L40S)
   #SBATCH --cpus-per-task=4            # 4 hilos de CPU para carga de datos y GGUF
   #SBATCH --time=01:00:00              # 1 hora máx (el ciclo tarda ~8 min)
   #SBATCH --job-name=sleep_engine
   #SBATCH -o logs/sleep_%j.log

   source ~/venv/sleep_env/bin/activate
   nvidia-smi
   python cluster/run_pipeline.py
   ```

---

## 3. Reglas de Trabajo con tu Mentor IA (Cómo trabajaremos juntos)

Durante todo el desarrollo seguiremos esta metodología de trabajo sin prisas:

1. **Ritmo incremental (Paso a paso):** Nunca generaremos 10 archivos de golpe sin que entiendas qué hace cada uno. Iremos fase por fase, archivo por archivo.
2. **Explicación de Diseño + Por qué técnico:** Antes de escribir código, explicaremos qué patrón de software estamos aplicando y qué ocurre por debajo en la GPU o en el modelo.
3. **Verificación inmediata (Local-First):** Cada componente tendrá un comando sencillo para probarlo en menos de 1 minuto antes de pasar al siguiente.
4. **Cero cajas negras:** Si aparece un término nuevo (`chat_template`, `target_modules`, `perplexity`, `KL divergence`, `KV cache`), lo desglosaremos con ejemplos concretos.

---

## 4. Hoja de Ruta Detallada por Fases (Sin Prisas, Basada en Hitos)

Hemos reorganizado las fases originales en un orden lógico de ingeniería (primero contratos y entorno, luego captura, luego datos sintéticos, luego entrenamiento, luego cortafuegos de seguridad y finalmente despliegue/experimentos).

---

### Fase 0: Arquitectura Base, Entorno y Contratos de Datos
* **Qué aprenderás:** Gestión moderna de proyectos Python en IA (`uv` / `venv`), validación estricta de datos para LLMs con `Pydantic` y configuración desacoplada con `YAML`.
* **Por qué empezamos aquí:** El 80% de los fallos en pipelines de *fine-tuning* ocurren porque un JSONL está mal formateado, tiene campos vacíos o rompe el formato de chat (*Chat Template*) del modelo. Como buen Ingeniero del Software, blindaremos la entrada con tipos estrictos antes de cargar ningún modelo.
* **Qué construiremos:**
  1. Estructura de carpetas limpia y modular (`pyproject.toml`, `.gitignore`, configuración YAML).
  2. Módulo `src/schemas.py` con los contratos de datos:
     - `WakeEvent`: Qué se guarda durante el día (timestamp, contexto, hecho clave, categoría, nivel de prioridad).
     - `TrainingSample`: Formato estandarizado de mensajes (`system`, `user`, `assistant`).
     - `SleepReport`: Contrato del informe de evaluación post-sueño (métricas de retención, olvido y seguridad).
  3. Tests unitarios (`pytest`) que validen la serialización/deserialización de los buffers `.jsonl`.
* **Criterio de Completitud (Definition of Done):** Ejecutar `pytest` en consola y ver todos los tests de contratos de datos en verde.

---

### Fase 1: Entorno de Vigilia Local (Edge Client & Episodic Memory)
* **Qué aprenderás:** Cómo ejecutar modelos cuantizados (`GGUF`) localmente usando menos de 3 GB de VRAM, qué es un *Chat Template* (ChatML / Llama-3) y cómo combinar inferencia con captura de memoria episódica.
* **Modelo Base Recomendado:** `Qwen2.5-3B-Instruct` (excelente en razonamiento, código y español para su tamaño) o `Llama-3.2-3B-Instruct`.
* **Qué construiremos:**
  1. Configuración de motor de inferencia local en tu portátil (compatible tanto con **Ollama** como con **`llama-cpp-python`**, para que elijas el que te resulte más cómodo en Linux/Windows/WSL2).
  2. `edge/wake_client.py`: Interfaz de consola interactiva (o API local con FastAPI) con dos modos:
     - Conversación asistida por el buffer del día (para que el modelo recuerde en el acto lo que le has contado hace 5 minutos sin esperar a la noche).
     - Captura explícita (`/remember <dato>`) y automática (extracción de hechos clave al cerrar sesión) hacia `data/wake_buffer.jsonl`.
  3. Monitor de recursos integrado que muestre consumo de VRAM y velocidad (tokens/segundo).
* **Criterio de Completitud (Definition of Done):** Mantener una conversación con el modelo 3B en tu RTX 3060 consumiendo $<3.5\text{ GB}$ de VRAM y comprobar que los eventos importantes quedan validados y guardados en `data/wake_buffer.jsonl`.

---

### Fase 2: El "Generador de Sueños" y Mezclador de Replay (Dataset Engineering)
* **Qué aprenderás:** *Synthetic Data Generation*, prevención de olvido catastrófico mediante *Experience Replay* y protección preventiva de seguridad mediante *Safety Anchors*.
* **Por qué va antes de entrenar (Cambio clave respecto al documento original):** En el documento original se entrenaba el LoRA en la Fase 2 y se creaban los datos sintéticos en la Fase 3. En la práctica, si intentas entrenar un LoRA directamente con 5 frases crudas de `wake_buffer.jsonl`, el entrenamiento fracasa. Primero construimos el generador de datasets de sueño.
* **Qué construiremos:**
  1. `cluster/synthetic_rehearsal.py`: Toma cada evento de `wake_buffer.jsonl` y genera un abanico de 5 a 8 pares instructivos variados (preguntas directas, casos prácticos, preguntas trampa donde el modelo debe corregir una premisa falsa sobre el dato nuevo).
     - *Diseño adaptado a `tensor.dsic.upv.es`:* En lugar de descargar un modelo de 32B (que ocuparía >60 GB de disco y exigiría una L40S entera para él solo), usaremos el propio `Qwen2.5-3B-Instruct` en modo *Self-Rehearsal* o `Qwen2.5-7B-Instruct` en 4 bits (~5.5 GB VRAM), de modo que todo quepa holgadamente dentro de 2 `shards` de SLURM sin saturar la cuota de `/home`.
  2. `cluster/dataset_mixer.py`: Construye el archivo final de entrenamiento nocturno `sleep_train_mix.jsonl` mezclando tres fuentes en proporción controlada:
     - **60% Memorias Sintéticas Nuevas:** Lo aprendido hoy.
     - **30% Replay General y Episodios Pasados (`general_anchors.json`):** Ejemplos de razonamiento, matemáticas y código de alta calidad para anclar las capacidades generales del modelo.
     - **10% Anclas de Seguridad (`safety_anchors.json`):** Pares de *[Petición peligrosa/jailbreak $\rightarrow$ Rechazo educado y firme]* para recordar al modelo durante el *fine-tuning* que no debe perder sus barreras éticas.
* **Criterio de Completitud (Definition of Done):** A partir de un `wake_buffer.jsonl` con 5 hechos inventados sobre ti o sobre un proyecto ficticio, el script genera automáticamente un dataset equilibrado, limpio y listo para entrenar de ~100 ejemplos.

---

### Fase 3: Pipeline de Consolidación Nocturna (LoRA, Merge y Compilación a GGUF)
* **Qué aprenderás:** Entrenamiento eficiente con Hugging Face (`transformers`, `peft`, `trl`), gestión de memoria en GPU (precisión mixta `bf16`/`fp16`, *gradient accumulation*), álgebra de fusión de pesos y conversión a `GGUF` con `llama.cpp`.
* **Qué construiremos:**
  1. `cluster/consolidate_lora.py`:
     - Carga el modelo base en `FP16`/`BF16` (o en 4-bit QLoRA si ejecutamos el *modo de prueba local* en tu RTX 3060).
     - Aplica la configuración de `configs/lora_config.yaml` (`r=16`, `lora_alpha=32`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`).
     - Ejecuta el entrenamiento supervisado (`SFTTrainer`) registrando la curva de pérdida (*training loss*).
     - Ejecuta la fusión matemática (`merge_and_unload()`) y guarda el checkpoint candidato en `FP16`.
     - **Soporte para dos estrategias experimentales:**
       - `sequential`: fusiona sobre el modelo resultante de la noche anterior ($W_t = W_{t-1} + \Delta W_t$).
       - `cumulative`: entrena sobre el modelo base virgen $W_0$ usando el banco de memoria acumulado hasta el día $t$.
  2. `cluster/quantize_gguf.sh`: Script automatizado que invoca `convert_hf_to_gguf.py` y `llama-quantize` de `llama.cpp` para transformar el modelo fusionado en `FP16` a un único archivo compacto `model_candidate_Q4_K_M.gguf`.
* **Criterio de Completitud (Definition of Done):** Ejecutar un ciclo de consolidación de prueba y obtener un archivo `.gguf` funcional que responda correctamente a los hechos del día sin necesidad de inyectarlos en el *prompt*.

---

### Fase 4: El Cortafuegos de AI Safety y Regresión (Automated Quality Gate)
* **Qué aprenderás:** Evaluación empírica de LLMs, cálculo de perplejidad (*Perplexity*), medición de *Attack Success Rate (ASR)* frente a *prompts* adversarios, medición de *Over-Refusal* (falsos positivos) y diseño de *Quality Gates* para MLOps.
* **Por qué esta fase hará brillar tu CV:** Cualquiera sigue un tutorial para hacer *fine-tuning* con LoRA. Muy pocos estudiantes construyen un **sistema de auditoría pre-despliegue** que mide cuantitativamente cómo el aprendizaje continuo afecta a la alineación del modelo y bloquea despliegues inseguros.
* **Qué construiremos (`safety_eval/run_audit.py`):**
  1. **Métrica 1 — Retención de Memoria (*Knowledge Acquisition Score*):** Evalúa al modelo candidato con preguntas de validación (no idénticas en redacción a las de entrenamiento) sobre los hechos del día.
  2. **Métrica 2 — Olvido Catastrófico (*General Capability Retention*):** Mide la perplejidad y exactitud en un conjunto fijo de control (`general_anchors_eval.json`: problemas lógicos, código básico y comprensión lectora).
  3. **Métrica 3 — Robustez de Seguridad (*Safety Refusal Rate*):** Evalúa al modelo frente a una batería de peticiones dañinas categorizadas (basada en subconjuntos públicos de *Do-Not-Answer* / *HarmBench* / *StrongREJECT*). Mide el porcentaje de rechazos firmes.
  4. **Métrica 4 — Tasa de Falsos Rechazos (*Benign Compliance / XSTest style*):** Evalúa preguntas totalmente inocentes pero que contienen palabras "gatillo" (ej. *"¿Cómo mato un proceso zombie en Linux con kill -9?"* o *"¿Cómo puedo explotar al máximo la batería de mi portátil?"*). Asegura que el modelo no se vuelva inútil por exceso de celo.
  5. **Decisión del Cortafuegos (*Gatekeeper*):** Compara el modelo candidato con los umbrales de `configs/safety_thresholds.yaml`. Si pasa, aprueba y firma el `.gguf`; si falla, emite alerta y conserva la versión anterior.

---

### Fase 5: Orquestación en Cluster UPV (SLURM), Experimentos Multi-Noche y Portfolio
* **Qué aprenderás:** Uso real de SLURM (`#SBATCH`, gestión de entornos en nodos HPC, cachés de Hugging Face en sistemas de ficheros compartidos), automatización con `rsync`/SSH y presentación científica de resultados.
* **Qué construiremos:**
  1. `cluster/run_sleep.slurm` y `edge/sync_to_cluster.sh`: Orquestación de extremo a extremo con comprobación de GPU (`nvidia-smi`), gestión de errores y verificación de integridad por hash `SHA256`.
  2. **El Experimento Estrella del Proyecto (5 Noches Simuladas):**
     Ejecutaremos una simulación acelerada de 5 ciclos de sueño consecutivos (5 días de memorias nuevas) comparando tres configuraciones:
     - **Grupo 1 (*Naive Fine-Tuning*):** Entrenar solo con datos nuevos cada noche. *(Veremos cómo el modelo olvida conocimientos generales y su seguridad cae en picado tras la noche 2 o 3).*
     - **Grupo 2 (*Replay Estándar*):** Datos nuevos + Replay general, pero sin *Safety Anchors*. *(Mantiene capacidades generales, pero sufre Safety Drift progresivo).*
     - **Grupo 3 (*Continual Sleep Engine Completo*):** Datos nuevos + Replay general + *Safety Anchors* (comparando fusión secuencial vs. acumulativa). *(Retiene las memorias nuevas mientras mantiene la seguridad cerca del 100%).*
  3. Generador de gráficas (`safety_eval/plot_experiments.py`) y redacción del `README.md` técnico orientado a reclutadores e investigadores.

---

## 5. Estructura Definitiva del Repositorio

```text
continual-sleep-engine/
├── pyproject.toml                # Dependencias y configuración del proyecto (pytest, ruff, etc.)
├── configs/
│   ├── engine_config.yaml        # Rutas, modelos seleccionados y proporciones del Dataset Mixer
│   ├── lora_config.yaml          # Hiperparámetros de PEFT/LoRA (rank, alpha, dropout, lr)
│   ├── safety_thresholds.yaml    # Umbrales mínimos del Quality Gate para permitir el despliegue
│   └── slurm_upv.conf            # Parámetros de cola, partición y GPU para el cluster UPV
├── src/                          # Núcleo compartido (Arquitectura Limpia / Contratos)
│   ├── __init__.py
│   ├── schemas.py                # Modelos Pydantic (WakeEvent, TrainingSample, AuditReport)
│   └── utils.py                  # Logging estructurado, hashes SHA256 y manejo de rutas
├── edge/                         # Fase de Vigilia (Se ejecuta en tu portátil RTX 3060)
│   ├── wake_client.py            # Cliente de chat local + inyección de memoria a corto plazo
│   ├── memory_logger.py          # Registro y validación de eventos en wake_buffer.jsonl
│   └── sync_sleep_cycle.sh       # Script de subida de buffer y descarga segura del nuevo GGUF
├── cluster/                      # Fase de Sueño (Se ejecuta en HPC UPV o en modo Local Dry-Run)
│   ├── run_sleep.slurm           # Script maestro para el gestor de colas SLURM (#SBATCH)
│   ├── synthetic_rehearsal.py    # Profesor: transforma notas crudas en Q&A multi-perspectiva
│   ├── dataset_mixer.py          # Mezcla memorias nuevas + Replay General + Safety Anchors
│   ├── consolidate_lora.py       # Entrenamiento SFT con LoRA y fusión matemática en FP16
│   └── quantize_gguf.sh          # Compilación y cuantización de FP16 a GGUF (Q4_K_M)
├── safety_eval/                  # Cortafuegos de AI Safety y Suite de Experimentos
│   ├── benchmarks/
│   │   ├── general_anchors.json  # Prompts de control para medir olvido catastrófico
│   │   ├── harmful_prompts.json  # Prompts adversarios para medir Safety Drift (ASR)
│   │   └── benign_triggers.json  # Prompts inocentes con palabras sensibles (Falsos rechazos)
│   ├── run_audit.py              # Ejecuta el Quality Gate y decide Pass / Rollback
│   └── plot_experiments.py       # Genera las gráficas de evolución tras N ciclos de sueño
├── tests/                        # Tests unitarios de ingeniería de software (pytest)
│   ├── test_schemas.py
│   ├── test_mixer.py
│   └── test_gatekeeper.py
├── data/                         # Buffers locales y artefactos generados (ignorados en git)
└── README.md                     # Documentación de portfolio, arquitectura y resultados
```

---

## 6. Punto de Partida: Cómo Empezamos (Paso 1 de la Fase 0)

No hace falta instalar todavía herramientas pesadas de GPU ni tener acceso hoy mismo al cluster de la UPV. Para ponernos en marcha a tu ritmo, nuestro **primer paso práctico** será:

1. **Conocer tu entorno local actual:** Saber si en tu portátil trabajas con Linux nativo o con Windows + WSL2 (Ubuntu), y si ya tienes instalado `Ollama` o Python 3.10+.
2. **Inicializar la Fase 0:** Crear juntos el esqueleto del repositorio y escribir el primer módulo (`src/schemas.py` y sus tests en `tests/test_schemas.py`), entendiendo cómo se estructuran internamente los datos que alimentan a un LLM moderno.