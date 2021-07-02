from matplotlib.backend_bases import MouseButton
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
import sys
import time
import logging


def array_in_arraylist(array, arraylist):
    for _i,  _array in enumerate(arraylist):
        if np.array_equal(array, _array):
            return True
    return False


def unique_arrays(lst: list):
    used = []
    _ = [used.append(x) for x in lst if not array_in_arraylist(x, used)]
    return used


class SegmentLabeler:

    def __init__(self,
                 img,
                 segmentation,
                 labeldict={'moho': 1, '410': 2, '660': 3, 'none': 9999},
                 loglevel=logging.INFO):
        """This class takes in an image, some corresponding 
        segmentation e.g. (SLIC) and a label dictionary that can be used to
        label the image in an 'intuitive' GUI.

        Examples
        --------

        The usage of this labeling is fairly straight forward. Given an image,
        ``img``, and a segmentation, ``segs``, of said image, we instantiate 
        the class and call the start labeling method.

        >>> from lwsspy.ml import SegmentLabeler
        >>> sl = SegmentLabeler(img, segments)
        >>> labeled_mask = sl.start_labeling()

        This will open two figures. One contains the image by itself and the 
        second one contains the image and the outlines of the mask. We will 
        use the firs image for reference and the second image to actually 
        label the image segments. The GUI has a few controls that can be used.

        ================= ======================================
        Control           Action
        ================= ======================================
        Mouse-left        Add label to segment
        Mouse-right       Remove label to segment
        Mouse-left-drag   Add labels to segments dragged over
        Mouse-right-drag  Remove labels to segments dragged over
        n                 Next label
        p                 Previous label  True
        d                 Delete previously labeled segment
        esc               Close figure and return the currently 
                          selected mask
        ================= ======================================

        The currently selected mask will also be returned if any of the figures 
        is closed.

        The selected mask can then be viewed via 

        >>> import matplotlib.pyplot as plt
        >>> imshow(labeled_mask, aspect='auto')

        Note that depending on the values you chose in the label dictionary
        you will have to create a colormap and norm that resembles the mask 
        values.


        Parameters
        ----------
        img : ndarray [w x h x 3]
            Image
        segmentation : ndarray [w x h]
            mask that has a unique nunmber for each segment such that it can be
            labeled.
        labeldict : dict, optional
            Dictionary of labels, must contain the 'none' keyword, which 
            denotes the unlabeled value, 
            by default {'moho': 1, '410': 2, '660': 3, 'none': 9999}
        loglevel : logging.LOGLEVEL, optional
            loglevel, used to debug the event loop. Not necessary to be 
            modified, by default logging.INFO


        Notes
        -----

        :Authors:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.07.02 00.00 (Lucas Sawade)

        """

        self.img = img
        self.segmentation = segmentation
        self.labeldict = labeldict

        self.safetyfirst()

        self.labeled = labeldict['none'] * np.ones_like(self.segmentation)

        # Pick variables
        self.pickhistory = dict()
        self.picklabels = []
        self.picknumber = []
        for label, number in self.labeldict.items():
            if label != 'none':
                self.pickhistory[label] = []
                self.picklabels.append(label)
                self.picknumber.append(number)
        self.activelabel = 0
        self.nlabels = len(self.picklabels)
        self.mouse_pressed = False

        # Logging
        self.loglevel = loglevel
        self.__setup_logger__()

    def safetyfirst(self):

        if 'none' not in self.labeldict:
            raise ValueError(
                "The label dictionary has to contain 'none' keywords.")

    def __setup_logger__(self):

        # create logger
        self.logger = logging.getLogger('SegmentationLabeler')
        self.logger.setLevel(self.loglevel)

        # create console handler and set level to debug
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s | %(levelname)8s: '
            '%(message)s (%(filename)s:%(lineno)d)',
            datefmt='%m/%d/%Y %I:%M:%S %p')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        if len(self.logger.handlers) > 0:
            self.logger.handlers = []

        self.logger.addHandler(ch)

        # 'application' code
        self.logger.debug('debug message')
        self.logger.info('info message')
        self.logger.warning('warn message')
        self.logger.error('error message')
        self.logger.critical('critical message')

    def plot_figure(self):

        self.logger.debug(f'Plotting Image & Segmentation...')
        self.static_figure = plt.figure(figsize=(6, 6))

        self.static_ax = plt.subplot(111)
        self.static_ax.imshow(self.img, aspect='auto')

        self.segment_figure = plt.figure(figsize=(6, 6))
        self.segment_ax = plt.subplot(111)
        self.segment_ax.imshow(
            mark_boundaries(self.img, self.segmentation, color=(0, 0, 0)),
            aspect='auto')

        self.plot_labeled_image()

    def plot_labeled_image(self):

        self.logger.debug(f'Plotting Labeled Image ...')

        # Get boundary color norm based on label numbers
        pickarray = np.array(self.picknumber)
        dpickarray = np.diff(pickarray)/2

        # Very artificial create color bounds
        self.bounds = list(pickarray[:-1] + dpickarray)
        self.bounds = [pickarray[0] - dpickarray[0]] + self.bounds
        self.bounds = self.bounds + [pickarray[-1] + dpickarray[-1]]

        # Create adhoc cmap and norm
        self.cmap = plt.get_cmap('rainbow').copy()
        self.cmap.set_bad('lightgray', alpha=0.0)

        self.norm = BoundaryNorm(self.bounds, self.cmap.N)

        # Get mask
        self.__update_labeled_image__()

    def start_labeling(self):

        self.plot_figure()
        self.segment_ax.set_title(
            f"Picking {self.picklabels[self.activelabel]}")
        plt.draw()
        self.segment_figure.canvas.mpl_connect('key_press_event', self.onkey)
        self.segment_figure.canvas.mpl_connect(
            'button_press_event', self.onclick)
        self.segment_figure.canvas.mpl_connect(
            'button_release_event', self.onrelease)
        self.segment_figure.canvas.mpl_connect(
            'motion_notify_event', self.onmotion)

        # Closing connections
        self.cidsegment = self.segment_figure.canvas.mpl_connect(
            'close_event', self.onclose_segment)
        self.cidstatic = self.static_figure.canvas.mpl_connect(
            'close_event', self.onclose_static)

        plt.show(block=True)

        return self.labeled

    def onclick(self, event):
        self.logger.debug(f'button press -- {event.button}')
        self.clicktime = time.time()
        self.mouse_pressed = True
        self.coords = []

    def onmotion(self, event):

        # duration = time.time() - self.clicktime

        # Second statement means nothing in toolbar selected
        if self.mouse_pressed \
                and self.segment_figure.canvas.manager.toolbar.mode == '':
            # and np.isclose((duration * 100) % 5, 0):
            # Get pixel locations
            x, y = int(event.xdata), int(event.ydata)
            self.logger.debug(f"Pick location:  ({x}, {y})")

            # Append coordinates
            self.coords.append((x, y))

    def onrelease(self, event):

        self.mouse_pressed = False
        self.logger.debug(f'button_release_event -- {event.button}')

        # Return if pick not in axes
        if event.inaxes != self.segment_ax:
            return

        # Return if pick  is happening while toolbar active
        state = self.segment_figure.canvas.manager.toolbar.mode
        if state == '':
            self.logger.debug(f"No figure tool selected.")
        else:
            self.logger.debug(f"{state} selected.")
            return

        duration = time.time() - self.clicktime

        # Get pixel locations
        x, y = int(event.xdata), int(event.ydata)

        self.logger.debug(f"Pick location:  ({x}, {y})")

        # Label single coordinate
        if event.button is MouseButton.LEFT and duration <= 0.3:
            self.logger.debug("Left Mouse button pressed")
            self.__label_segment__(x, y)

        # Label many coordinates
        elif event.button is MouseButton.LEFT and duration > 0.3:
            self.logger.debug("Left Mouse button dragged")
            for x, y in self.coords:
                self.__update_labeled__(y, x)
            self.__update_labeled_image__()

        # Remove single coordinates
        elif event.button is MouseButton.RIGHT and duration <= 0.3:
            self.logger.debug("Right Mouse button pressed")
            self.__remove_segment__(x, y)

        # Remove many coordinates
        elif event.button is MouseButton.RIGHT and duration > 0.3:
            self.logger.debug("Right Mouse button dragged")
            for x, y in self.coords:
                self.__reset_segment__(x, y)
            self.__update_labeled_image__()

        # Remove duplicates
        self.pickhistory[self.picklabels[self.activelabel]] = \
            unique_arrays(
                self.pickhistory[self.picklabels[self.activelabel]])

    def __label_segment__(self, x, y):

        # Update labeled and then update the image plotted on top
        self.__update_labeled__(y, x)
        self.__update_labeled_image__()

    def __remove_segment__(self, x, y):

        #  Update labeled and the update image
        self.__reset_segment__(x, y)
        self.__update_labeled_image__()

    def __reset_segment__(self, x, y):

        # Get value of segmented image
        val = self.segmentation[y, x]

        # Find where everything is that is in the segment
        pos = np.where(np.isclose(self.segmentation, val))

        # Remove the entry from the history
        for _i,  _histpos in \
                enumerate(self.pickhistory[self.picklabels[self.activelabel]]):

            if np.array_equal(_histpos, pos):
                self.pickhistory[self.picklabels[self.activelabel]].pop(_i)
                break

        # Reset the labeled array
        self.labeled[pos] = self.labeldict['none']

    def __update_labeled__(self, x, y):

        # Get value of segmented image
        val = self.segmentation[x, y]

        # Find where everything is that is in the segment
        pos = np.where(np.isclose(self.segmentation, val))

        # Add pos to history
        self.pickhistory[self.picklabels[self.activelabel]].append(pos)

        # Set Labeled to active label number
        self.labeled[pos] = self.picknumber[self.activelabel]

    def onkey(self, event):
        self.logger.debug(f'key press -- {event.key}')

        if event.key == 'n':
            self.__next_label__()

        elif event.key == 'p':
            self.__previous_label__()

        elif event.key == 'd':
            self.__remove_previous__()

        elif event.key == 'escape':
            self.__stop_labeling__()

    def __next_label__(self):

        self.activelabel += 1

        if self.activelabel == self.nlabels:
            self.activelabel -= 1

        self.segment_ax.set_title(
            f"Picking {self.picklabels[self.activelabel].capitalize()}")

        self.segment_figure.canvas.draw()

    def __previous_label__(self):

        self.activelabel -= 1

        if self.activelabel == -1:
            self.activelabel += 1

        self.segment_ax.set_title(
            f"Picking {self.picklabels[self.activelabel]}")

        self.segment_figure.canvas.draw()

    def __remove_previous__(self):
        self.logger.debug(
            f"Remove previous selection for label -- "
            f"{self.picklabels[self.activelabel]}")

        if len(self.pickhistory[self.picklabels[self.activelabel]]) == 0:
            return

        # Get last position
        pos = self.pickhistory[self.picklabels[self.activelabel]][-1]

        # Reset labeled to the  masked value
        self.labeled[pos] = self.labeldict['none']

        # Pop latest one
        self.pickhistory[self.picklabels[self.activelabel]].pop(-1)

        # Update the  labeled image
        self.__update_labeled_image__()

    def __update_labeled_image__(self):

        self.logger.debug(f'Updating Labeled Image ...')

        self.labeled_m = np.ma.masked_values(
            self.labeled, self.labeldict['none']
        )

        if hasattr(self, 'labeled_img'):
            self.labeled_img.set_data(self.labeled_m)
        else:
            self.labeled_img = self.segment_ax.imshow(
                self.labeled_m, cmap=self.cmap, norm=self.norm, aspect='auto',
                alpha=0.5
            )

        self.segment_figure.canvas.draw()

    def onclose_segment(self, event):
        self.logger.debug("Closing Segmentation Figure")

        self.static_figure.canvas.mpl_disconnect(self.cidstatic)

        self.__stop_labeling__()

    def onclose_static(self, event):
        self.logger.debug("Closing Static Figure")

        self.segment_figure.canvas.mpl_disconnect(self.cidsegment)

        self.__stop_labeling__()

    def __stop_labeling__(self):
        self.logger.debug("Stop Labeling & quitting the program")

        fignums = plt.get_fignums()
        if len(fignums) != 0:
            for num in fignums:
                plt.close(num)

        # Removed the plotted image so that it can be plotted again
        if hasattr(self, 'labeled_img'):
            del self.labeled_img

        # Stop logging
