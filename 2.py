# importing librarys 
from skimage import color, data, io, filters, feature, viewer, util 
import matplotlib.pyplot as plt
from skimage import exposure, restoration, morphology 
import skimage 

def show_image(image, title='Image', cmap_type='gray'):
    plt.figure(figsize=(16, 12))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
def plot_comparison(original, filtered, title_filtered):
    fig, (ax1, ax2) = \
        plt.subplots(ncols=2, figsize=(16, 12), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis('off')
    
group_image = io.imread(r'Resources\Photos\1.jpg') 
show_image(group_image, 'Group image')

grayscale_group_image = color.rgb2gray(group_image) 
show_image(grayscale_group_image, 'Grayscale Group Image') 

#Gaussian Blue 
grayscale_group_image_blurred = filters.gaussian(group_image, sigma=1) 
show_image(grayscale_group_image_blurred,'Blurred Image')

#Noise
noisy_group = util.random_noise(group_image)
plot_comparison(group_image, noisy_group, 'Noisy Group')

#Smoothened 
group_smooth = filters.gaussian(noisy_group, multichannel=True)
plot_comparison(noisy_group, group_smooth, 'Smoothened')   

denoised_group = restoration.denoise_bilateral(noisy_group, multichannel=True)
plot_comparison(noisy_group, denoised_group, 'Denoised Group') 

#Improve image quality
group_image_adapteq = exposure.equalize_adapthist(group_image, clip_limit=.01)  
plot_comparison(group_image, group_image_adapteq, 'Adaptive Equalized')

# Sobel 
edges = filters.sobel(grayscale_group_image)  
show_image(edges,'Edges Image') 

group_edges_1 = feature.canny(grayscale_group_image)
show_image(group_edges_1,'Edge1 Image') 

group_edges_2 = feature.canny(grayscale_group_image, sigma=2) 
plot_comparison(group_edges_1,group_edges_2, 'Edge2 Image')




