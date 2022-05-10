from model.gan import GAN

if __name__ == "__main__":
    gan = GAN() #TODO: pass in constructor parameters
    gan.gan_training_loop()