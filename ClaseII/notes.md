# Clase 2

> Nota: se recomienda usar la extensión Latex Workshop para VSCode.

## Transformers Decoders

**Espacio latente**: espacio de información que consta de las características de los datos como la relación entre palabras.

### Generative Pretrained Transformer (GPT)

Paper: [Improving Language Understanding by Generative Pre-Training][paper-improving-language-understanding-by-gpt]

Introduce un framework que logra un buen entendimiento del lenguaje (natural) mediante un modelo utilizando `generative pre-training` y `discriminative fine-tuning`.

#### Framework

Consta de dos etapas:

1. Aprender un modelo de lenguaje de gran capacidad a partir de un amplio corpus de texto.
1. Etapa de `fine-tuning`, en donde se adapta el modelo a una tarea discriminativa con datos etiquetados.

**Etapa 1 - Unsupervised pre-training**
Maximiza una función likelihood (ver paper, pág. 3, sección 3.1), en donde:

- `k`: es el tamaño de la *ventana de contexto*.
- `P`: es la probabilidad condicional modelada utilizando una Red Neuronal con parámetros `0 - theta`.
- `0 - theta`: parámetros entrenados mediante *stochastic gradient descent*.

Utilizan multi-layer Transformer decoder, que es una variante del transformer.
Este modelo aplica una operación de *multi-head self-attention* sobre los input context tokens, seguido de una capa feedforward por posición (position-wise) para generar una distribución sobre los target tokens.

> El objetivo del preentrenamiento es predecir la siguiente palabra de una secuencia, con la meta de aprender y estructurar patrones de lenguaje natural.

**Etapa 2 - Supervised fine tuning**
Una vez entrenado el modelo según la función likelihood, se adaptan los parámetros a la tarea de *supervisión*.

Se encontró que incluir el modelo de lenguaje como un objetivo auxiliar en el fine-tuning ayudó a:

- Mejorar la generalización del modelo.
- Acelerar la convergencia.

**Task-specific input transformations**
En algunas tareas, como la clasificación de texto, se puede aplicar el fine-tuning explicado anteriormente.
Otras requieren modificaciones al nivel de tareas, como por ejemplo, responder preguntas (tríos de *documentos, preguntas y respuestas*).

- *Textual entailment* (implicación textual): Se concatenan la premisa `p` y la hipótesis `h` como secuencias de tokens, con un token delimitador (`$`) entre ellas.
- *Questions answering and commonsens reasoning*: Se da un documento de contexto `z`, una pregunta `q`, un set de posibles respuestas ${a_k}$. Se concatenan `z` y `q` con cada posible respuesta agregando un token delimitador $[z;q;\$;a_k]$. Cada secuencia se procesa independientemente con el modelo y se normaliza mediante la capa `softmax` para producir una distribución de salida sobre las posibles respuestas.

#### Attention types

Sitio: [Understanding and Coding Self-Attention, Multi-Head Attention, Causal-Attention, and Cross-Attention in LLMs][site-understanding-attention]

Todo surge para solucionar algunos problemas que tienen las *Recurrent Nueral Networks* (RNN), por ejemplo, la traducción de una oración de un idioma a otro. Con RNN esto se haría palabra a palabra, lo cual no es útil para el problema en cuestión.

Los mecanismos de atención aparecen entonces para dar accesso a todos los elementos de la secuencia en cada paso.

Los mecanismos de **self-attention** le permiten al modelo *pesar* la importancia de cada elemento en la secuencia de entrada, y dinámicamente ajuster su *influencia* en el output.

> Note: self-attention es como se conoce en el paper original a la operación scaled-dot product, el cual sigue siendo el mecanismo más utilizado.

**Self-attention**
Utiliza tres matrices de peso: $W_q$, $W_k$ y $W_v$, que se ajustan como parámetros del modelo durante el *training*.
Permiten proyectar los inputs a *queries*, *keys* y *values* componentes de la secuencia.

- **Query**: The query is a representation of the current word used to score against all the other words (using their keys). We only care about the query of the token we’re currently processing.
- **Key**: Key vectors are like labels for all the words in the segment. They’re what we match against in our search for relevant words.
- **Value**: Value vectors are actual word representations, once we’ve scored how relevant each word is, these are the values we add up to represent the current word.

Se obtienen al hacer multiplicación matricial entre la matriz de pesos $W$ y los inputs embeddings $x$.

- Query sequence: $q^{(i)} = x^{(i)}W_q$ por cada $i$ en la secuencia $1...T$.
- Key sequence: $k^{(i)} = x^{(i)}W_k$ por cada $i$ en la secuencia $1...T$.
- Values sequence: $v^{(i)} = x^{(i)}W_v$ por cada $i$ en la secuencia $1...T$.

Los vectores $q^{(i)}$ y $k^{(i)}$ tienen dimension $d_k$. Las matrices de proyección $W_q$ y $W_k$ tienen forma $d x d_k$, mientras que $W_v$ tiene $d x d_v$.

Aquí se computa el producto-punto de los vectores *query* y *key*, por lo que tienen que tener el mismo tamaño, es decir, $d_k$ = $d_q$. La cantidad de elementos de $v^{(i)}$, que determina el tamaño del vector de contexto resultante, puede ser arbitraria.

Una vez obtenidas las key y values, se procede a computar los pesos de atención no normalizados (*unnormalized attention weights*) u `ω (omega)`, donde:

$\omega_{i,j} = q^{(i)} k^{(j)}$.

Luego se *normaliza* los pesos de atención, obteniendo $\alpha$, al aplicar la función **softmax**. Adicionalmente, $1/\sqrt{d_k}$ se usa para escalar $\omega$ antes de normalizar.

$a_n = softmax{\frac{w_n}{\sqrt{d_k}}}$

Finalmente, se calcula el Vector de contexto $z^{(i)}$.

$z^{(i)} = \sum^{T}_{j=1} \alpha_{i, j} v^{(j)}$

**Por qué el escalado mediante raíz cuadrada?**
Se asegura que la longitud euclídea de los vectores de peso tengan la misma magnitud. Esto ayuda a que los pesos de atención de volverse muy pequeño o grande, lo cual puede llevar a inestabilidad numérica o afectar la convergencia del modelo durante el *training*.
El producto-punto de q y k es la suma de $d_k$ términos independientes, cada uno con varianza $1$. Por esto, la varianza del score crece linealmente con $d_k$. Al divirdlo por $\sqrt{d_k}$ se cancela ese crecimiento y la **varianza vuelve a 1**.

**Multi-head attention**
Sea un *single attention head* las matrices $W_q$, $W_k$ y $W_v$, se considera *multi-head attention* al conjunto de varias de estas. Sería el equivalente análogo al uso de múltiples *kernels* en una red convolucional.

La ventaja es que cada una de estas *heads* puede potencialmente enfocarse en aprender diferentes partes de la secuencia de input. También se beneficia en términos de ejecución al utilizar computación paralela.

> Nota: el paso *forward* implica aplicar cada capa de SelfAttention al input de manera independiente. Luego, los resultados se concatenan en la última dimensión (dim=-1).

**Causal self-attention**
También conocido como *masked self-attention*, se asegura que los outputs para cierta posición en una secuencia se basen solamente en los outputs de posiciones previas y no de de posiciones futuras.
Básicamente, la predicción de la próxima palabra solo depende de las palabras anteriores.

Por ello, se *enmascaran* los token sucesivos al token actual.

*Optimización*
Dados los *attention scores*, se enmascaran los valores por encima de la diagonal con $-\infty$ antes de enviar los valores a la *softmax* para computar los *attention weights*.

### Supervised fine tuning

Source: [Understanding and using supervised fine tuning][site-understanding-and-using-supervised-fine-tuning].

SFT recopila un dataset con resultados de alta calidad generados por LLMs, sobre los que el modelo se ajusta utilizando un objetivo estándar de modelo de lenguaje.

**RLHF**: Reinforcement learning from human feedback.

**Alignment**: un modelo de lenguaje preentrenado no es generalmente útil. Si se generan outputs con este modelo, los resultados pueden ser repetitivos y poco útiles. Para mejorarlo es que se lo debe *alinear* (aligment o *to align*) siguiendo un "framework" de tres pasos: **pre-training** -> **SFT** -> **RLHF**.

***What is SFT?***

Consiste en recolectar un dataset de alta calidad correspondiente a la salida de un LLM. Luego, se aplica fine-tunning en el modelo con estos datos. Es "supervisado" porque el dataset contiene casos/ejemplos de como se debe comportar el modelo.

### Generative Pretrained Transformers (GPT-2)

Source: [Ilustrated GPT 2][site-ilustrated-gpt2].

***GPT-2***

- Arquitectura similar a *decoder-only* transformer.
- Entrenado con un dataset masivo.
- GPT2 uses Byte Pair Encoding to create the tokens in its vocabulary. This means the tokens are usually parts of words.
- At training time, the model would be trained against longer sequences of text and processing multiple tokens at once. Also at training time, the model would process larger batch sizes (512) vs. the batch size of one that evaluation uses.

GPT-2 uses **transformer decoder** blocks, while BERT uses **transformer encoder** blocks.

Cómo funciona? Una vez que se produce un token, este se agrega a la secuencia de inputs. Luego, esta secuencia se convierte en el input del modelo en el próximo paso. Esto se conoce como ***auto-regression***.

BERT usa self-attention mientras que GPT-2 utiliza ***masked self-attentions***.

### GPT-3: Language Models are Few-Shot Learners

Source: [GPT-3: Language Models are Few-Shot Learners][site-language-models-are-few-shot-learners]

- GPT-3 no utilizo fine-tuning, en su lugar utilizó Few-shot, One-shot, Zero-shot.

### Instruct GPT

Source: [Instruction Following][site-openai-instruct-gpt].

***GPT-3***

- Las LLMs pueden generar contenido tóxico, falso y no util para el usuario.

***Instruct GPT***

- surge para corregir las debilidades de GPT-3 mediante fine-tuning y reinforcement-learning.
- To make our models safer, more helpful, and more aligned, we use an existing technique called reinforcement learning from human feedback (RLHF)⁠.
- The resulting InstructGPT models are much better at following instructions than GPT‑3.
- They also make up facts less often, and show small decreases in toxic output generation.
- InstructGPT utiliza un reward model con los outputs preferidos por humanos, luego se optimiza el reward model mediante Proximal Policy Optimization (PPO).

***Methods***

1. Collect demonstration data and train a supervised policy.
1. Collect comparison data, and train a reward model.
1. Optimize a policy against the reward model using reinforcement learning.

<!-- Links -->
<!-- Papers -->
[paper-improving-language-understanding-by-gpt]: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
<!-- Sites -->
[site-understanding-attention]: https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
[site-understanding-and-using-supervised-fine-tuning]: https://cameronrwolfe.substack.com/p/understanding-and-using-supervised
[site-ilustrated-gpt2]: https://jalammar.github.io/illustrated-gpt2/
[site-language-models-are-few-shot-learners]: https://sh-tsang.medium.com/review-gpt-3-language-models-are-few-shot-learners-ff3e63da944d
[site-openai-instruct-gpt]: https://openai.com/index/instruction-following/
