# Clase 4

> Nota: se recomienda usar la extensión Latex Workshop para VSCode.

## MoEs, prompting y evaluación

Moe o **Mixture of Experts** consiste en un conjunto de redes neuronales donde cada una aprende a manejar un subconjunto específico de información.

C/experto recibe el mismo input y genera la misma cantidad de outputs, pero existe un mecanismo de selección que controla qué expertos contribuyen a la predicción final.

$P_j$: probabilidad de que se elija un experto específio *j*.

Una red neuronal llamada gating network realiza la selección determinando las contribuciones de cada experto. La gating network aprende las contribuciones de cada experto basado en el input.

**Esparcidad**
Source: [Huggingface - MoE - Sparsity][site-moe-sparsity].

En un modelo denso, todos los parámetros se usan para todos los inputs, mientras que en un modelo esparzo se utilizan parcialmente los parámetros.

**MoE en LLM**
Source: [Huggingface - MoE][site-moe].

Permite a los modelos ser preentrenados con mucho menor cómputo, lo cual permite escalar el modelo o el tamaño de dataset con el mismo *presupuesto* de cómputo necesario para un modelo denso.
Un modelo que implementa MoE debería alcanzar la misma calidad que uno denso de manera mucho más rápida durante el *pre-entrenamiento*.

### Componentes

- **Capa MoE esparza**: Las capas MoE suelen tener cierta cantidad de *expertos* (e.g. 8), donde cada uno es una red neuronal y se usan en lugar de capas FNN. En la práctica, los *expertos* son **feed-forward networks** (***FNN***) layers, o incluso otros MoEs (lo que lleva a *hierarchical MoEs*).
- **Gate Network o router**: determina qué tokens son enviados a qué experto. Un token puede ser enviado a más de un experto. El **router** es compensado por los parámetros aprendidos y es ***pre-entrenado*** al mismo tiempo que el resto de la red.

### Desafíos

- Entrenamiento: si bien los MoE permiten tener un pretraining más eficiente en cuanto a cómputo, tienen problemas de generalización durante *fine-tuning*, lo cual lleva al *overfitting*.
- Inferencia: si bien los MoE tienen muchos parámetros, solo algunos de ellos se utilizan durante la inferencia. Esto lleva a su vez a una inferenca mucho más rápida comparada con un modelo denso con la misma cantidad de parámetros. PERO, todos los parámetros tienen que ser cargados en la RAM, es decir, tiene mayores requisitos de memoria.

<!-- links -->
[site-moe]: https://huggingface.co/blog/moe
[site-moe-sparsity]: https://huggingface.co/blog/moe#what-is-sparsity
