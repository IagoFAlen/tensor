# Tensor

Glossário:
RANK    => Quantidade de dimensões que o Tensor possui.
SHAPE   => Tamanho de cada dimensão no Tensor de N RANK.
STRIDES => Quantidades de passos necessários para que a dimensão mude.

## Blocos contíguos de memória.
A forma mais eficiente de inserir valores dentro dos blocos de memória acontece quando esses valores são inseridos diretamente um após o outro. Isso porque, dentro da CPU, podemos usar a função memset considerando o início do bloco, o vetor dos dados e o tamanho da inserção.
No entanto, isso só pode acontecer se o Tensor que estamos lidando possui RANK 1, ou seja, é um Tensor de apenas uma dimensão.

### CPU
#### Por que 1D é um bloco contíguo?
Vamos observar a estrutura de um Tensor 1D de tamanho K na memória.

No exemplo a seguir, K possui o tamanho 10.
```
  i  =   0     1     2     3     4     5     6     7     8     9     
T[K] = [0x00, 0x01, 0X02, 0X03, 0X04, 0x05, 0x06, 0X07, 0X08, 0X09]
```

Agora, queremos inserir em T[] os seguintes pesos:
```
W[K] = [0.65f, 0.50f, 0.21f, 0.98f, 0.01f, 0.24f, 0.42f, 0.10f, 0.78f, 0.71f]
```

Mas vamos além disso, queremos que essa inserção aconteça por particionamento. Suponde que cada partição tem tamanho 2, podemos ter:
```
 i =   0      1
W_p1[0.65f, 0.50f]
W_p2[0.21f, 0.98f]
W_p3[0.01f, 0.24f]
W_p4[0.42f, 0.10f]
W_p5[0.78f, 0.71f]
```

Aqui fica a chave do bloco contíguo:
Quando vamos passar os sub-blocos W_p para T[i], cada sequência é adjacente. Isso implica que
```
T[0] = W_p1[0]
T[1] = W_p1[1]
T[2] = W_p2[0]
T[3] = W_p2[1]
T[4] = W_p3[0]
T[5] = W_p3[1]
T[6] = W_p4[0]
T[7] = W_p4[1]
T[8] = W_p5[0]
T[9] = W_p5[1]
```

Se observarmos o endereço de memória de cada um deles, observar que cada peso está um do lado do outro.
```
&W_p1[0] = [0x00]
&W_p1[1] = [0x01]
&W_p2[0] = [0x02]
&W_p2[1] = [0x03]
&W_p3[0] = [0x04]
&W_p3[1] = [0x05]
&W_p4[0] = [0x06]
&W_p4[1] = [0x07]
&W_p5[0] = [0x08]
&W_p5[1] = [0x09]
```

#### Por que RANKS maiores que 1 não são contíguos?
Porque RANKS maiores usam STRIDES para a inserção. A adjacência só acontece logicamente, não fisicamente. Vou explicar com mais detalhes:
Para apresentar essa abordagem, usaremos um Tensor de RANK 2 com tamanho 9 (L x C). Sendo L = 3 e C = 3, ou seja, SHAPE {3, 3}.

Fisicamente o mapeamento do Tensor é um array linear 1D:
```
  i  =   0     1     2     3     4     5     6     7     8       
T[K] = [0x00, 0x01, 0X02, 0X03, 0X04, 0x05, 0x06, 0X07, 0X08]
```

Logicamente, o mapeamento do Tensor é uma matriz L x C:
```
          T[L x C]

        0     1     2     
    0 [0x00, 0x01, 0X02] 
    1 [0X03, 0X04, 0x05]
    2 [0x06, 0X07, 0X08]
```

Usamos a estratégia row-major para percorrer o vetor com o STRIDE. Isso significa que o cálculo do STRIDE se dá por:
```
    stride[rank - 1] = 1;
    for (size_t i = rank - 1; i-- > 0;)
        stride[i] = stride[i + 1] * shape[i + 1];
```

Isso se traduz em:
STRIDE = {3, 1}

Ou seja, precisamos andar um bloco de memória para mudar de coluna, precisamos andar 3 colunas para caminhar uma linha.

O jeito mais eficiente de inserir no Tensor é usando o mapeamento do STRIDE porque assim comportamos qualquer quantidade de dimensões dinamicamente.

Dessa vez, queremos inserir em T[] os seguintes pesos:
```
W[K] = [0.05f, 0.08f, 0.09f, 0.13f, 0.15f, 0.18f, 0.21f, 0.24f, 0.29f]
```

Seguindo a ideia de particionamento do bloco podemos criar os sub-blocos como:
```
 i =   0      1      2
W_p1[0.05f, 0.08f, 0.09f]
W_p2[0.13f, 0.15f, 0.18f]
W_p3[0.21f, 0.24f, 0.29f]
```

Diferente de uma dimensão, pularemos as casas com o STRIDE, e aqui está a diferença mais importante:

Em vez de ser assim:
```
T[0] = W_p1[0]
T[1] = W_p1[1]
T[2] = W_p1[2]
T[3] = W_p2[0]
T[4] = W_p2[1]
T[5] = W_p2[2]
T[6] = W_p3[0]
T[7] = W_p3[1]
T[8] = W_p3[2]
```

É assim:
```
T[0] = W_p1[0]
T[1] = W_p2[0]
T[2] = W_p3[0]
T[3] = W_p1[1]
T[4] = W_p2[1]
T[5] = W_p3[1]
T[6] = W_p1[2]
T[7] = W_p2[2]
T[8] = W_p3[2]
```

ou seja
```
&W_p1[0] = [0x00]
&W_p1[1] = [0x03]
&W_p1[2] = [0x06]

&W_p2[0] = [0x01]
&W_p2[1] = [0x04]
&W_p2[2] = [0x07]

&W_p3[0] = [0x02]
&W_p3[1] = [0x05]
&W_p3[2] = [0x08]
```
