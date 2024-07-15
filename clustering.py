import os
import sys
import numpy as np
from PIL import Image
#import kmeans
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.decomposition import PCA


html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Test Page</title>
    </head>
    <body>
    """


# read png files from the directory
def read_files(dirname):
    for file in os.listdir(dirname):
        if file.endswith('.png'):
            image = Image.open(f'{dirname}/{file}')
            image = image.convert('L')
            yield (np.array(image), file)


def display_image(image):
    return Image.fromarray(image)

def display_images(images, dirname):
    # create a directory to store the images
    os.makedirs(dirname, exist_ok=True)
    for i, image in enumerate(images):
        image = display_image(image)
        image.save(f'{dirname}/{i}.png')

def rescale_all_in_set(images, new_shape):
    return [resize(image, new_shape) for image in images]

def rescale_set(images):
    new_shape = (32, 32)
    return rescale_all_in_set(images, new_shape)

def size_clustering(images):
    # for each image get its dimensionality
    dimensionalities = [image.shape for image in images]

    # cluster the images based on the dimensionality using DBSCAN
    clustering = DBSCAN(eps=1, min_samples=2).fit(dimensionalities)
    return clustering

def image_clustering(images, n_clusters):
    images = np.array(rescale_set(images))
    n_samples, width, height = images.shape
    images = images.reshape((n_samples, width * height))
    pca = PCA(n_components=0.95, svd_solver='full')
    images = pca.fit_transform(images)
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(images)
    return clustering

def display_clustered_images(images, clustering, dirname, imagesdir):
    # create html files to display the images in the browser

    html_files = []
    files_names = []

    
    for cluster in range(clustering.n_clusters):
        
        files_names.append([])
        # create new html file for each cluster
        html_files.append(open(f'{dirname}_{cluster}.html', 'w'))
        html_files[-1].write(html_content)


    for i, cluster in enumerate(clustering.labels_):
        files_names[cluster].append(images[1][i])
        html_files[cluster].write(f'<img src="{imagesdir}/{images[1][i]}">')
        html_files[cluster].write(f'<HR>')

    
    for i in range(clustering.n_clusters):
        html_files[i].write('</body></html>')
        html_files[i].close()

    return files_names

def display_clustered_images_DBSCAN(images, clustering, dirname, imagesdir):
    # assuming that the clustering is DBSCAN
    # create html files to display the images in the browser
    
    html_files = []
    files_names = []

    for cluster in set(clustering.labels_):
        
        files_names.append([])
        # create new html file for each cluster
        html_files.append(open(f'{dirname}_{cluster}.html', 'w'))
        html_files[-1].write(html_content)
   
    
    for i, cluster in enumerate(clustering.labels_):
        files_names[cluster].append(images[1][i])
        html_files[cluster].write(f'<img src="{imagesdir}/{images[1][i]}">')
        html_files[cluster].write(f'<HR>')

    for i in set(clustering.labels_):
        html_files[i].write('</body></html>')
        html_files[i].close()

    return files_names

def write_to_output_file(names):
    for name in names:
        with open('output.txt', 'a') as f:
            f.write(f'{name} ')
    
    with open('output.txt', 'a') as f:
        f.write('\n')
    

def main():

    if len(sys.argv) < 2:
        print("Usage: python script.py <argument>")
        sys.exit(1)

    dirname = sys.argv[1]
    # lambda to calculate if the image is small
    is_small = lambda x: (x.shape[0] <= 4 or x.shape[1] <= 4) or (x.shape[0] <= 6 and x.shape[1] <= 6)
    is_big   = lambda x: x.shape[0] > 20 or x.shape[1] > 20
    # lambda to calculate ratio of demension of the image
    ratio    = lambda x:  x.shape[0] / x.shape[1]
    # cluster images based on the ratio of the dimension of the image

    # create empty file output.txt
    open('output.txt', 'w').close()


    small_images   = ([], [])
    merged_letters = ([], [])
    thick_images   = ([], [])
    normall_images = ([], [])
    thin_images    = ([], [])
    tall_images    = ([], [])


    THICK_THRESHOLD = 0.63
    NORMALL_THRESHOLD = 0.75
    THIN_THRESHOLD = 1.3
    TALL_THRESHOLD = 1.8

    for image, imagename in read_files(dirname):
        
        if is_small(image):
            small_images[0].append(image)
            small_images[1].append(imagename)
        elif ratio(image) < THICK_THRESHOLD or is_big(image):     
            merged_letters[0].append(image)         # [0, THICK_THRESHOLD]
            merged_letters[1].append(imagename)
        elif ratio(image) < NORMALL_THRESHOLD:   
            thick_images[0].append(image)           # (THICK_THRESHOLD, NORMALL_THRESHOLD)
            thick_images[1].append(imagename)
        elif ratio(image) <= THIN_THRESHOLD:      
            normall_images[0].append(image)         # (NORMALL_THRESHOLD, THIN_THRESHOLD)
            normall_images[1].append(imagename)
        elif ratio(image) < TALL_THRESHOLD:
            thin_images[0].append(image)            # (THIN_THRESHOLD, TALL_THRESHOLD)
            thin_images[1].append(imagename)
        else:
            tall_images[0].append(image)            # (TALL_THRESHOLD, inf)
            tall_images[1].append(imagename)

    small_clusterring = size_clustering(small_images[0])

    thick_clusterring   = image_clustering(thick_images[0], 4)
    normall_clusterring = image_clustering(normall_images[0], 9)
    thin_clusterring    = image_clustering(thin_images[0], 10)
    tall_clusterring    = image_clustering(tall_images[0], 4)

    small_images_clusterlist =  display_clustered_images_DBSCAN(small_images, small_clusterring, 'small_images_clustered', dirname)
    thick_images_clusterlist =  display_clustered_images(thick_images, thick_clusterring, 'thick_images_clustered', dirname)
    normall_images_clusterlist = display_clustered_images(normall_images, normall_clusterring, 'normall_images_clustered', dirname)
    thin_images_clusterlist = display_clustered_images(thin_images, thin_clusterring, 'thin_images_clustered', dirname)
    tall_images_clusterlist = display_clustered_images(tall_images, tall_clusterring, 'tall_images_clustered', dirname)

    for clusterlist in small_images_clusterlist:
        write_to_output_file(clusterlist)
   
    for clusterlist in thick_images_clusterlist:
        write_to_output_file(clusterlist)
    
    for clusterlist in normall_images_clusterlist:
        write_to_output_file(clusterlist)
    
    for clusterlist in thin_images_clusterlist:
        write_to_output_file(clusterlist)
  
    for clusterlist in tall_images_clusterlist:
        write_to_output_file(clusterlist)
    
    write_to_output_file([merged_letters[1]])

    merged_html = open('merged_letters.html', 'w')
    merged_html.write(html_content)
    for imagename in merged_letters[1]:
        merged_html.write(f'<img src="{dirname}/{imagename}">')
        merged_html.write(f'<HR>')
    merged_html.write('</body></html>')
    

if __name__ == '__main__':
    main()

                  