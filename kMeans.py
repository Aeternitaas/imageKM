import numpy as np
from scipy import misc

# Defines which troup the certain datapoint identifies with.
def closest_centroids( points, centroids ):
    """
    Finds nearest centroid pixel. Get differences between each point
    and all centroids (elementwise subtraction)

    Returns:
        an array containing the index to the nearest centroid for each point
    """

    # Each row here will be an array of differences between a
    # Each entry in the matrix of pixels will be subtracted by where the centroids are.
    # Find distances of each pixel to the centroids and find nearest.
    differences = points - centroids[:, np.newaxis, np.newaxis]

    # Square every value of differences; distance formula.
    squareDifferences = differences**2

    # Sum rgb differences (x1-c1 => sum(x1-c1))
    # Give me one distance for each point
    # axis = 3; converts to 2d
    summedSquareDifferences = squareDifferences.sum( axis = 3 )

    # square root distances to finish distance formula
    finalDistances = np.sqrt( summedSquareDifferences )

    # return vector of which centroids are closest
    # return a matrix of colours grouped up together; simply finds the
    # closest centroid and sets the value
    return np.argmin( finalDistances, axis = 0 )

# Moves the centroids to average; redefines.
def move_centroids( points, closest, centroids ):
    """
    Returns:
        the new centroids assigned from the points closest to them
    """

    newCentroids = []
    
    # iterate over all k
    for i in range( centroids.shape[0] ):

        # find which points are in group i
        # returns a matrix of t/f for all points
        indices = ( closest == i ) 

        # gets all points that are in group i
        correspondingPoints = points[ indices ] 
    
        # get the average rgb values for the points
        average = correspondingPoints.mean( axis = 0 )

        # add it to our new centroids list
        newCentroids.append( average )

    # return our new centroids; return a numpy array
    return np.array( newCentroids )

def initialize_centroids( points, k ):
    """
    Grabs k random pixels and return them.

    Args:
        points: image matrix
        k: # of iterations

    Returns:
        k centroids from the initial points
    """

    ( x, y, z ) = points.shape
    # take that 3d into 2d
    centroids = points.copy().reshape( x * y, z )
    np.random.shuffle( centroids )
    # return the first k items in the list
    return centroids[:k]

def set_to_centroids( points, centroids, closestCentroids ):
    """
    Sets like-pixels to their nearest centroids.

    Returns:
        matrix of all points set to the value of their corresponding centroids
    """

    # Make a matriox to hold new points
    newPoints = np.zeros( points.shape )

    # Set all points to corresponding centroids
    (x, y) = closestCentroids.shape
    for i in range( x ):
        for j in range( y ):
            newPoints[i][j] = centroids[ closestCentroids[i][j] ]
    
    # newPoints = centroids[ closestCentroids ] ]
    return newPoints

def kMeans( points, k, maxIter = 10):
    """
    Returns:
        k means clustered points
    """

    # init centroids randomly
    centroids = initialize_centroids( points, k )

    # iterate as many times as we should
    for i in range( 0, maxIter ):
        print( "Iteration: " + str( i + 1 ) )
        # groups nearby similar pixels in an RGB matrix with the randomly chosen centroids
        closestCentroids = closest_centroids( points, centroids ) 
        # sets all the pixels in that range as the randomly chosen colour
        centroids = move_centroids( points, closestCentroids, centroids )

    # return our new matrix
    finalPoints = set_to_centroids( points, centroids, closestCentroids )
    return finalPoints

def main():
    # get input
    imageName = input( "Please enter image name: ")
    points = misc.imread( imageName ) # W x H x 3 Array
    k = int( input( "Please enter k: ")) # number of groups

    # get new matrix, save it away
    newPoints = kMeans( points, k )

    # save new pixels to a new image
    misc.imsave( "outfile.jpg", newPoints )

if __name__ == "__main__":
    main()
