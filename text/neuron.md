## Perceptron

$ z = w_0 x_0 + w_1 x_1 + ... + w_n x_n = \textbf{w}^T \textbf{x} $

$
\phi(z) =
\begin{cases}
1, & z \geq \theta \\
-1, & \text{otherwise}
\end{cases}
$

$ w_0 = \text{bias} = - \theta $

$ x_0 = 1 $

### Learning rate

$ \Delta w_j = \eta (y^{(i)} - \hat y^{(i)}) x^{(i)}_j $
$ \hat y^{(i)} $ is the predicted weight.

Only converges when the classes can be linearly separable and the learning rate is sufficiently small.

## ADAptative LInear NEurom (Adaline)

Ground for:

- logistic regression;
- support vector machines;
- regression models.

$ \phi(\textbf{w}^T \textbf{x}) = \textbf{w}^T \textbf{x} $

### Cost function

$ J(\textbf{w}) = \frac{1}{2} \sum_{i} (y^{(i)} - \phi (z^{(i)}))^2 $

To find the minimum, we get the vector opposite to the gradient vector:

$$
\begin{align*}
-\frac{\partial J}{\partial w_j} & = -\sum_{i} (y^{(i)} - \phi (z^{(i)})) \frac{\partial}{\partial w_j} \phi (z^{(i)}) \\
& = -\sum_{i} (y^{(i)} - \phi (z^{(i)})) x^{(i)}_j \\
\\
\Delta \textbf{w} & = -\eta \sum_{i} (y^{(i)} - \phi (z^{(i)})) \textbf{x}^{(i)}
\end{align*}
$$

The new weight is computed considering all samples in the training set (called Batch Gradient Descent or BGD).

## Hyperparameters and SGD

- Learning rate ($ \eta $);
- Number of epochs (`n_iter`).

### SGD

Learning from the whole training data takes too long.
We can use one sample for each iteration instead.
That model is called Stochastic Gradient Descent or SGD.
You can also use SGD with mini-batches instead of a single sample each iteration.
You should also shuffle the dataset every epoch to prevent cycles.

Learning rate in SGD is also replaced by an adaptive learning rate that decreases over time, such as:
$ \frac{c_1}{\text{n iterations} + c_2} $, $c_1$, $c_2$ constants.

#### Mini-batches

Mini-batches can also be useful because you can use vectorized operations for the batches instead of the for loop, which can be accelerated with parallel computations.

### Online learning

Model is trained as data arrives (e.g. web apps).
In these cases, the training data can be discarded and the system can adapt immediately as new data arrives.
SGD can be used in this case.

## Sigmoids

### Logistic regression

$ \phi(z) = \frac{1}{1 + e^{-z}} $

$
\hat y =
\begin{cases}
1, & \phi(z) \geq 0 \\
0, & \text{otherwise}
\end{cases}
$

### Hyperbolic tangent

$ \phi(z) = \frac{1 - e^{-2z}}{1 + e^{-2z}} $

$
\hat y =
\begin{cases}
1, & \phi(z) \geq 0 \\
-1, & \text{otherwise}
\end{cases}
$

Activation function adds non-linearity to the system.

### Multiple layers

$ xW = b $

### Gradient of sigmoid

By definition, sigmoid comes from the logistic differential equation:

$$ \frac{d}{dx} \sigma(x) = \sigma(x) (1 - \sigma(x)) $$
$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

Then, the gradient is:

$$
\begin{align*}
\nabla \sigma(z) & = \sigma(z) (1 - \sigma(z)) \frac{d}{d \textbf{w}} z \\
& = \sigma(z) (1 - \sigma(z)) \textbf{x}
\end{align*}
$$

$$ \Delta \textbf{w} = -\eta \sigma(z) (1 - \sigma(z)) \textbf{x} $$

### ReLU and gradient

$$ \sigma (z) = max(0, z) $$

ReLU gradient is the Heaviside step function, that means it doesn't dissipate values, so it converges faster.

## Propagation

Forward propagation: The process where the network receives input, processes it through all of its layers until it goes through its output layer.

Backward propagation: The process where we compare the results with the expected output, compute the error and send that error back trying to minimize it.
BP algorithms determine the loss at the output and propagate it back.
We do so by determining the gradient of each node leading to the expected value and then we compute the new weights based on it.

One epoch is one round of forward and back propagation.

## Multilayer perceptron (MLP)

Consist of at least three layers of nodes (the first being only the input and the other two being neuron layers).
They use backpropagation for training and can be used to solve non-linear problems.

### Methodology of Neural Network

Steps:

1. Read input and output;
2. Initialize weights and biases matrices with random data (or use some method);
3. Calculate hidden layer input;
4. Perform nonlinear transformations;
5. Repeat in all following layers (including output layer);
6. Compute gradient of error at the output layer;
7. Compute slopes at all layers;
8. Compute delta at output layer (from slope);
9. Compute error comparing delta with layer values;
10. Repeat in all previous layers;
11. Update weights and biases using computed deltas.
