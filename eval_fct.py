import numpy as np
import theano
import theano.tensor as T

from sampler_fct import binary_sample


def reconstruct_images(true_x, num_steps, params, energy, srng, fraction=0.7, D=784):
    """
    Noises up fraction of each images and reconstructs it via axis aligned optimization
    """

    # Randomly noise up fraction of the image
    noise_n = int(D * fraction)
    noise_ind = T.argsort(srng.uniform(size=true_x.shape), axis=1)
    noise_ind = T.sort(noise_ind[:, :noise_n])
    ind = T.arange(true_x.shape[0]).reshape((-1, 1))
    ind = T.repeat(ind, noise_n, axis=1)
    fills = binary_sample(size=(true_x.shape[0], noise_n), srng=srng)
    fake_x = T.set_subtensor(true_x[ind.flatten(),
                                    noise_ind.flatten()],
                             fills.flatten())

    # Run axis aligned optimization
    def step(i, x, *args):
        x_i = x[T.arange(x.shape[0]), i]
        x_reversed = T.set_subtensor(x_i, 1.0 - x_i)
        merged = T.concatenate([x, x_reversed], axis=0)

        eng = energy(merged).flatten()
        eng_x = eng[:x.shape[0]]
        eng_r = eng[x.shape[0]:]
        cond = T.gt(eng_x, eng_r)
        # The update values
        updated = T.switch(cond, x_i, 1.0 - x_i)
        return T.set_subtensor(x_i, updated)

    for i in range(num_steps):
        shuffle = srng.uniform(noise_ind.shape)
        shuffle_ind = T.argsort(shuffle, axis=1)

        shuffled = noise_ind[ind.flatten(), shuffle_ind.flatten()]
        shuffled = T.reshape(shuffled, noise_ind.shape)

        result, _ = theano.scan(fn=step,
                                sequences=shuffled.T,
                                outputs_info=fake_x,
                                non_sequences=params)
        fake_x = result[-1]

    # Incorrectly look at the whole images
    correct_pixels = T.mean(T.cast(T.eq(true_x, fake_x), theano.config.floatX))
    # Get the ratio of correctly predicted pixels
    #     bad_pixels = true_x[ind.flatten(), noise_ind.flatten()]
    #     reconstruct_pixels = fake_x[ind.flatten(), noise_ind.flatten()]
    #     correct_pixels = T.mean(T.cast(T.eq(bad_pixels, reconstruct_pixels), theano.config.floatX))
    return fake_x, correct_pixels
