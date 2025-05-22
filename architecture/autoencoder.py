# Convolutional Autoencoder
# Based on https://blog.keras.io/building-autoencoders-in-keras.html

latent_dim = 32   #(X,X,latent_dim)
SIZE = 64         #(SIZE,SIZE,3)

# Autoencoder
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([                                # 64x64x3
      layers.Conv2D(128, (3, 3), activation='relu', padding='same'),    # 64x64x16
      layers.MaxPooling2D((2, 2), padding='same'),                      # 32x32x16
  
      layers.GroupNormalization(groups=-1),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same'),         # 32x32x8
      layers.MaxPooling2D((2, 2), padding='same'),                          # 16x16x8

      layers.GroupNormalization(groups=-1),
      layers.Conv2D(latent_dim, (3, 3), activation='relu', padding='same'),    # 16x16x8
      layers.MaxPooling2D((2, 2), padding='same'),                             # 8x8x8
    ])
    self.decoder = tf.keras.Sequential([
      layers.Conv2D(latent_dim, (3, 3), activation='relu', padding='same'),    # 8x8x8
      layers.UpSampling2D((2, 2)),                                             # 16x16x8

      layers.BatchNormalization(),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same'),         # 16x16x8
      layers.UpSampling2D((2, 2)),                                          # 32x32x8

      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), activation='relu', padding='same'),    # 32x32x16
      layers.UpSampling2D((2, 2)),                                      # 64x64x16
      layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')    #64x64x3

    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
