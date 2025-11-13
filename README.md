# Tensor

## Glossário
- **RANK**: Quantidade de dimensões do tensor. Ex.: Um tensor 2D (matriz) tem rank 2.
- **SHAPE**: Tamanho de cada dimensão. Ex.: shape = {3, 4} para uma matriz 3x4.
- **STRIDES**: Quantidade de elementos a pular na memória para avançar uma posição em cada dimensão. Ex.: Para shape = {3, 4} em row-major, strides = {4, 1}.

## Blocos Contíguos de Memória
A forma mais eficiente de inserir valores em um tensor é quando os dados são copiados sequencialmente, sem lacunas na memória, usando funções como memcpy (CPU) ou cudaMemcpy (GPU). Isso é possível porque essas funções aproveitam otimizações de hardware, como acesso à cache e transferência em blocos.

Um bloco é contíguo se seus elementos ocupam posições consecutivas na memória (array weights). Isso ocorre em:
- Tensores 1D (rank = 1), onde todos os elementos são naturalmente contíguos.
- Tensores ND (rank > 1), quando o bloco cobre toda a última dimensão (ou várias dimensões internas), respeitando os strides. Assim, com última dimensão estamos falando dos saltos que stride[0] dá. Se eu bloco de memória não preenche a dimensão completamente, o próximo particionamento será inserido num local não padronizado, ou seja, não contíguo.

### CPU
#### Por Que 1D é um Bloco Contíguo?
Um tensor 1D é armazenado como um array linear na memória, onde cada elemento está adjacente ao próximo. Considere um tensor 1D com tamanho K = 10:
```
i    =   0     1     2     3     4     5     6     7     8     9
T[K] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]
```

Queremos inserir os pesos:
```
W[K] = [0.65f, 0.50f, 0.21f, 0.98f, 0.01f, 0.24f, 0.42f, 0.10f, 0.78f, 0.71f]
```

Podemos particionar em sub-blocos de tamanho 2:
```
i    =   0      1
W_p1 = [0.65f, 0.50f]
W_p2 = [0.21f, 0.98f]
W_p3 = [0.01f, 0.24f]
W_p4 = [0.42f, 0.10f]
W_p5 = [0.78f, 0.71f]
```

Cada sub-bloco é copiado sequencialmente:
```
T[0] = W_p1[0] // 0.65f
T[1] = W_p1[1] // 0.50f
T[2] = W_p2[0] // 0.21f
T[3] = W_p2[1] // 0.98f
T[4] = W_p3[0] // 0.01f
...
T[9] = W_p5[1] // 0.71f
```

Os endereços de memória são consecutivos:
```
&T[0] = [0x00]
&T[1] = [0x01]
&T[2] = [0x02]
...
&T[9] = [0x09]
```
Isso é contíguo porque não há pulos na memória, e podemos usar um único memcpy(T, W, K * sizeof(float)) para copiar tudo.

#### Por Que Blocos em Tensores ND (Rank > 1) Podem Não Ser Contíguos?
Embora o tensor seja armazenado como um array 1D linear (weights), a organização multi-dimensional (definida por shape e strides) faz com que os elementos de um bloco ND possam estar espalhados na memória, dependendo do block_shape.

Em row-major (como no código), a última dimensão é contígua (stride[rank-1] = 1), mas as dimensões externas têm strides maiores, introduzindo pulos. Um bloco é contíguo apenas se cobrir toda a última dimensão (ou várias dims internas).

##### Exemplo: Tensor 2D
Considere um tensor 2D com shape = {3, 3} (3 linhas, 3 colunas). O array linear é:
```
i    =   0     1     2     3     4     5     6     7     8
T[K] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
```
Logicamente (row-major):
```
        T[L x C]
      0     1     2
0 [0x00, 0x01, 0x02] // Linha 0
1 [0x03, 0x04, 0x05] // Linha 1
2 [0x06, 0x07, 0x08] // Linha 2
```

Os strides são calculados como:
```
stride[rank - 1] = 1; // Última dimensão (colunas)
stride[i] = stride[i + 1] * shape[i + 1];
Para shape = {3, 3}:
- stride[1] = 1 (colunas contíguas).
- stride[0] = 1 * 3 = 3 (avançar uma linha pula 3 elementos).
```

Queremos inserir os pesos:
```
W[K] = [0.05f, 0.08f, 0.09f, 0.13f, 0.15f, 0.18f, 0.21f, 0.24f, 0.29f]
```

Particionamos em sub-blocos (cada um é uma linha):
```
i    =   0      1      2
W_p1 = [0.05f, 0.08f, 0.09f] // Linha 0
W_p2 = [0.13f, 0.15f, 0.18f] // Linha 1
W_p3 = [0.21f, 0.24f, 0.29f] // Linha 2
```

O mapeamento é:
```
T[0] = W_p1[0] // 0.05f
T[1] = W_p1[1] // 0.08f
T[2] = W_p1[2] // 0.09f
T[3] = W_p2[0] // 0.13f
T[4] = W_p2[1] // 0.15f
T[5] = W_p2[2] // 0.18f
T[6] = W_p3[0] // 0.21f
T[7] = W_p3[1] // 0.24f
T[8] = W_p3[2] // 0.29f
```

Endereços:
```
&T[0] = [0x00] // W_p1[0]
&T[1] = [0x01] // W_p1[1]
&T[2] = [0x02] // W_p1[2]
&T[3] = [0x03] // W_p2[0]
&T[4] = [0x04] // W_p2[1]
&T[5] = [0x05] // W_p2[2]
&T[6] = [0x06] // W_p3[0]
&T[7] = [0x07] // W_p3[1]
&T[8] = [0x08] // W_p3[2]
```

##### Contiguidade
- Bloco Contíguo: O bloco inteiro block_shape = {3, 3} é contíguo, pois cobre toda a última dimensão (shape[1] = 3). Os índices [0, 1, 2, 3, 4, 5, 6, 7, 8] são consecutivos, permitindo:
memcpy(T, W, 9 * sizeof(float));
- Bloco Não Contíguo: Um bloco menor, como block_shape = {2, 2} (sub-matriz 2x2), tem índices não consecutivos:
  - Para start_coord = {0, 0}:
    - [0,0] → 0*3 + 0*1 = 0
    - [0,1] → 0*3 + 1*1 = 1
    - [1,0] → 1*3 + 0*1 = 3
    - [1,1] → 1*3 + 1*1 = 4
  - Índices: [0, 1, 3, 4] (pulo entre 1 e 3 devido a stride[0] = 3).
  - Não contíguo, mas divisível em dois segmentos contíguos: [0, 1] (linha 0) e [3, 4] (linha 1), cada um copiado com memcpy.

##### Por Que Não Contíguo?
Os strides das dimensões externas (e.g., stride[0] = 3) criam pulos na memória entre elementos de um bloco, a menos que o bloco cubra toda a última dimensão. Em row-major, a última dimensão é contígua, então blocos que a preenchem inteiramente (e.g., uma linha completa em 2D) são contíguos.

#### Inserção Eficiente
- Contíguo: Use um único memcpy para o bloco inteiro.
- Não Contíguo: Divida o bloco em segmentos contíguos (cada um cobrindo a última dimensão) e use memcpy para cada segmento, ajustando offsets com strides.
- O uso de strides permite manipular qualquer rank dinamicamente, mapeando coordenadas ND para índices 1D com:
idx = sum(coords[i] * stride[i])

## Threads
