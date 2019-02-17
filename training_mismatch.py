import numpy, keras, matplotlib.pyplot as plt

class VisMismatch(keras.callbacks.Callback):
    def from_imagenet(self, image):
        return (image + 1.) / 2.
    def __init__(self, val_gen, cols=2, rows=2, limit = 100, fig_size = 3, preprocess_input = None,
                 viz_type=None, single_input_channels=3):
        self.cols = cols
        self.rows = rows
        self.limit = limit
        self.fig_size = fig_size
        self.val_gen = val_gen
        self.batch_counter = 0
        self.preprocess_input = 'imagenet' if preprocess_input is not None else None
        self.viz_type = viz_type
        self.single_input_channels = single_input_channels

    def on_train_begin(self, logs={}):
        self.fig, self.axs =plt.subplots(self.rows, self.cols,figsize=(self.fig_size*self.cols,self.fig_size*self.rows))
        plt.ion()
        plt.draw()

    def on_batch_end(self, epoch, logs={}):
        self.batch_counter += 1
        if self.batch_counter % 200 != 1:
            return
        #self.losses.append(logs.get('loss'))
        x_input = None
        y_true = None
        y_pred = None

        for (x,y) in self.val_gen:
            y_pr = self.model.predict(x)
            if isinstance(x, list):
                x = x[0]
            if x_input is None:
                x_input = x
            else:
                x_input = numpy.concatenate((x_input, x), axis=0)
            if y_true is None:
                y_true = y
            else:
                y_true = numpy.concatenate((y_true, y), axis=0)
            if y_pred is None:
                y_pred = y_pr
            else:
                y_pred = numpy.concatenate((y_pred, y_pr), axis=0)
            if len(y_true) > self.limit: break
        if (len(y_true.shape) == 1): #binary
            incorrects = numpy.equal(y_true, numpy.around(numpy.squeeze(y_pred, axis=-1)))
        else:
            incorrects = numpy.equal(numpy.argmax(y_true, axis=-1), numpy.argmax(y_pred, axis=-1))
        y_wrong_ = numpy.where(incorrects == False)[0]
        plt.gcf().canvas.set_window_title("{}/{}".format(len(y_wrong_),len(y_true)))
        #plt.close()
        #self.fig.clear()
        #plt.cla()
        id_to_label = {v: k for k, v in self.val_gen.class_indices.items()}
        for i in range(1, self.rows * self.cols + 1):
            #self.fig.add_subplot(cols, cols, i)

            if i <= len(y_wrong_):
                image = x_input[y_wrong_[i - 1]].copy()
                if self.preprocess_input is not None:
                    image = self.from_imagenet(image)


                #for 3d
                if self.viz_type == 'series':
                    if len(image.shape) == 4:
                        image = numpy.concatenate((image[0],image[-1]), axis=1) #stacked first and last image, i.e. [5,500,500,3]
                    else:
                        if image.shape[2] == self.single_input_channels * 2: #i.e we have two images one underneath another as input (3x2 channels)
                            image = numpy.concatenate((image[:,:,:self.single_input_channels], image[:,:,self.single_input_channels:]), axis=1) #concat, i.e. [500,500,3x5]




                self.axs[(i-1) // self.cols, (i-1) % self.cols].imshow(image)
                if (len(y_true.shape) == 1):  # binary
                    label_indx = int(round(y_pred[y_wrong_[i - 1]][0]))
                else:
                    label_indx = numpy.argmax(y_pred[y_wrong_[i - 1]])
                self.axs[(i - 1) // self.cols, (i - 1) % self.cols].set_title("{}".format(id_to_label[label_indx]))
            else:
                self.axs[(i - 1) // self.cols, (i - 1) % self.cols].cla()
        plt.draw()
        plt.pause(.00001)
        #print(incorrects)