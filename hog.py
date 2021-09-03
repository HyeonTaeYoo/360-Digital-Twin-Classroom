
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os

def _hog_channel_gradient(channel):
    g_row = np.empty(channel.shape, dtype=np.float_)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]

    g_col = np.empty(channel.shape, dtype=np.float_)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    return g_row, g_col


# Calculation of the cell's HOG value
def cell_hog(magnitude,
             orientation,
             orientation_start, orientation_end,
             cell_columns, cell_rows,
             column_index, row_index,
             size_columns, size_rows,
             range_rows_start, range_rows_stop,
             range_columns_start, range_columns_stop):
    total = 0.

    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if cell_row_index < 0 or cell_row_index >= size_rows:
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)


# Generate line pixel coordinates
def _line_pix_coordinates(r0, c0, r1, c1):
    # r0, c0 - starting position (row, column)
    # r1, c1 - end position (row, column)
    steep = 0
    r = r0
    c = c0
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)

    rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)   # intp - integer used for indexing
    cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)

    if (c1 - c) > 0:
        sc = 1
    else:
        sc = -1
    if (r1 - r) > 0:
        sr = 1
    else:
        sr = -1
    if dr > dc:
        steep = 1
        c, r = r, c
        dc, dr = dr, dc
        sc, sr = sr, sc
    d = (2 * dr) - dc

    for i in range(dc):
        if steep:
            rr[i] = c
            cc[i] = r
        else:
            rr[i] = r
            cc[i] = c
        while d >= 0:
            r = r + sr
            d = d - (2 * dc)
        c = c + sc
        d = d + (2 * dr)

    # rr, cc - Indices of pixels that belong to the line.
    rr[dc] = r1
    cc[dc] = c1

    return np.asarray(rr), np.asarray(cc)


def vis_hog(hog_value, number_of_orientations,
            size_rows, size_columns,
            cell_rows, cell_columns,
            number_of_cells_rows, number_of_cells_columns):
    radius = min(cell_rows, cell_columns) // 2 - 1
    orientations_arr = np.arange(number_of_orientations)
    orientation_bin_midpoints = (np.pi * (orientations_arr + .5) / number_of_orientations)

    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((size_rows, size_columns), dtype=np.float_)

    for r in range(number_of_cells_rows):
        for c in range(number_of_cells_columns):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * cell_rows + cell_rows // 2,
                                c * cell_columns + cell_columns // 2])

                rr, cc = _line_pix_coordinates(int(centre[0] - dc),
                                               int(centre[1] + dr),
                                               int(centre[0] + dc),
                                               int(centre[1] - dr))
                hog_image[rr, cc] += hog_value[r, c, o]
    return hog_image


def hog(img, number_of_orientations, pixels_per_cell):
    size_rows, size_columns = img.shape
    cell_rows, cell_columns = pixels_per_cell
    number_of_cells_rows = int(size_rows // cell_rows)  # number of cells along row-axis
    number_of_cells_columns = int(size_columns // cell_columns)  # number of cells along col-axis

    gradient_rows, gradient_columns = _hog_channel_gradient(img)  # compute gradient
    magnitude = np.hypot(gradient_columns, gradient_rows)
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180

    r_0 = int(cell_rows / 2)
    c_0 = int(cell_columns / 2)
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    range_rows_stop = int(cell_rows / 2)
    range_rows_start = -range_rows_stop
    range_columns_stop = int(cell_columns / 2)
    range_columns_start = -range_columns_stop
    number_of_orientations_per_180 = int(180. / number_of_orientations)

    hog_value = np.zeros((number_of_cells_columns, number_of_cells_rows, number_of_orientations))

    # compute orientations integral images
    for i in range(number_of_orientations):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        # c = c_0
        r = r_0
        r_i = 0
        # c_i = 0

        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                hog_value[r_i, c_i, i] = \
                    cell_hog(magnitude, orientation,
                             orientation_start, orientation_end,
                             cell_columns, cell_rows, c, r,
                             size_columns, size_rows,
                             range_rows_start, range_rows_stop,
                             range_columns_start, range_columns_stop)
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows

    # generate HOG image for visualization
    hog_image = vis_hog(hog_value, number_of_orientations,
                        size_rows, size_columns,
                        cell_rows, cell_columns,
                        number_of_cells_rows, number_of_cells_columns)

    return hog_value, hog_image


def main():
    bins = 8
    pixels_per_cell = (8, 8)
    hog_feat_des_male = np.zeros((30, 28 * 28 * 8))
    hog_feat_des_female = np.zeros((30, 28 * 28 * 8))
    filenames = glob('D:/1INHA/Python/hair-segmentation/3d_hair_binary/3d_hair/*.jpg')
    
    for ind, name in enumerate(filenames):  # image files
        print(name)
        img = plt.imread(os.path.join(os.getcwd(), name)).astype(float)  # read image intensities
        print(img.shape)

        if len(img.shape) == 3:  # color image
            img = np.mean(img, axis=2)
            print(img.shape)
        r_max=0
        r_min=0
        c=0
        c_max=0
        
        rows, cols=img.shape
        for i in range(rows):
            if c>c_max:
                c_max=c;
            c=0
            for j in range(cols):
                if img[i][j]!=0:
                    c+=1
        
        for i in range(cols):
            for j in range(rows):
                if img[j][i]!=0:
                    if j>r_max:
                        r_max=j
                    if j<r_min:
                        r_min=j
        hog_value, hog_image = hog(img, bins, pixels_per_cell)
        print(hog_value.shape, hog_image.shape)
        if float(r_max/c_max)>0 and float(r_max/c_max)<1.5:
            hog_feat_des_male[ind, :] = hog_value.ravel()
        else:
            hog_feat_des_female[ind, :] = hog_value.ravel()


        # show input and HOG image side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.axis('off')
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Input image')

        ax2.axis('off')
        ax2.imshow(hog_image, cmap='gray')
        ax2.set_title('Histogram of Oriented Gradients')
        #plt.show()

    np.savetxt('output_male.txt', hog_feat_des_male, fmt='%.8f')
    np.savetxt('output_female.txt', hog_feat_des_female, fmt='%.8f')


if __name__ == '__main__':
    main()