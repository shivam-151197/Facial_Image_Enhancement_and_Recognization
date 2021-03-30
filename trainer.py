from Gan import Generator, Discriminator
from Data import Data
import random
import time
import os
import tensorflow as tf
from tensorflow.keras import layers, Input, losses
import matplotlib.pyplot as plt
path = 'img_align_celeba/'
build_discriminator = Discriminator()
discriminator = build_discriminator.Discriminator()
build_generator = Generator()
generator = build_generator.Generator()

###
#Change As per Your Computation Power
###
epochs = 10
batch_size =300
scale_factor =2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cross_entropy = losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



def show_image(low_img, generated_img):
	fig = plt.figure(figsize=(10, 7))
	fig.add_subplot(1, 2, 1)
	plt.imshow(generated_img, cmap = 'RGB')
	plt.axis('off')
	plt.title("Genrated")
	plt.imshow(low_img, cmap = 'RGB')
	plt.axis('off')
	plt.title('Input Image')

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step(input_batch, output_batch):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(input_batch, training=True)

      real_output = discriminator(output_batch, training=True)
      fake_output = discriminator(generated_images, training=True)
      img_show = random.randint(0, batch_size)
      show_image(real_images[img_show], generated_images[img_show])
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def train(real_images, input_images, epochs):
  for epoch in range(epochs):
    start = time.time()
    for i in range(0, len(real_images), batch_size):
      train_step(input_images[i:i+batch_size], real_images[i:i+batch_size])

    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

data = Data(path, scale_factor, 300)
real_images, input_images = data.collect()
train(real_images, input_images, epochs)