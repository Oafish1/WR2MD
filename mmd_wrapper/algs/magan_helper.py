from .source.magan.train import *


def magan_helper(xb1,
                 xb2,
                 labels1=None,
                 labels2=None,
                 batch_size=100,
                 rec_step=100,
                 max_iterations=100000,
                 correspondence_loss=correspondence_loss,
                 verbose=1,
                 plot=False,
                 ):
    """
    Implements MAGAN

    Parameters
    ----------
    xb1, xb2: np.array
        Datasets to align
    labels1, labels2: np.array
        Corresponding labels
    batch_size: int
        Batch size for GAN training
    rec_step: int
        How often to record statistics into the 'history' dictionary
    max_iterations: int
        Number of iterations to run
    verbose: int
        Whether or not to print updates on training
    plot: bool
        Whether or not to plot progress each rec_step.  Only works
        with labels

    Returns
    -------
    [
        [Mapped Datasets],
        Dictionary object with 'loss_D', 'loss_G', 'iteration'
    ]
    """

    # Change: Reusability
    tf.reset_default_graph()

    # Change: Add history dict for statistic storage
    history = {'loss_D': [],
               'loss_G': [],
               'iteration': []
               }

    if plot:
        plt.ion()
        fig = plt.figure()

    # Change: Function handler, no longer need generation
    # xb1, xb2, labels1, labels2 = get_data()

    # Change: Verbosity
    if verbose:
        print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1.shape, xb2.shape))

    # Prepare the loaders
    loadb1 = Loader(xb1, labels=labels1, shuffle=True)
    loadb2 = Loader(xb2, labels=labels2, shuffle=True)

    # Build the tf graph
    magan = MAGAN(dim_b1=xb1.shape[1], dim_b2=xb2.shape[1], correspondence_loss=correspondence_loss)

    # Train
    for i in range(1, max_iterations+2):
        # Change: Verbosity
        if verbose:
            if i % rec_step == 0:
                print("Iter {} ({})".format(i, now()))

        # Change: Optional labels
        xb1_ = loadb1.next_batch(batch_size)
        xb2_ = loadb2.next_batch(batch_size)
        if labels1 is not None:
            xb1_, labels1_ = xb1_
            xb2_, labels2_ = xb2_

        magan.train(xb1_, xb2_)

        # Evaluate the loss and plot
        if i % rec_step == 0:
            # Change: Optional labels
            xb1_ = loadb1.next_batch(10 * batch_size)
            xb2_ = loadb2.next_batch(10 * batch_size)
            if labels1 is not None:
                xb1_, labels1_ = xb1_
                xb2_, labels2_ = xb2_

            lstring = magan.get_loss(xb1_, xb2_)
            # Change: Verbosity
            if verbose:
                print("{} {}".format(magan.get_loss_names(), lstring))

            # Change: Recording
            history['iteration'].append(i-1)
            split_loss = lstring.split()
            history['loss_D'].append(float(split_loss[0]))
            history['loss_G'].append(float(split_loss[1]))

            # Change: Recording
            """
            history['xb1'].append(xb1)
            history['xb2'].append(xb2)
            history['Gb1'].append(Gb1)
            history['Gb2'].append(Gb2)
            """

            # Change: Verbosity
            if plot:
                xb1 = magan.get_layer(xb1_, xb2_, 'xb1')
                xb2 = magan.get_layer(xb1_, xb2_, 'xb2')
                Gb1 = magan.get_layer(xb1_, xb2_, 'Gb1')
                Gb2 = magan.get_layer(xb1_, xb2_, 'Gb2')

                fig.clf()
                axes = fig.subplots(2, 2, sharex=True, sharey=True)
                axes[0, 0].set_title('Original')
                axes[0, 1].set_title('Generated')
                axes[0, 0].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[0, 0].scatter(0,0, s=100, c='w'); axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
                axes[0, 1].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[0, 1].scatter(0,0, s=100, c='w'); axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
                axes[1, 0].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[1, 0].scatter(0,0, s=100, c='w'); axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
                axes[1, 1].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[1, 1].scatter(0,0, s=100, c='w'); axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);

                for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
                    axes[0, 0].scatter(xb1[labels1_ == lab, 0], xb1[labels1_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='b', marker=marker)
                    axes[0, 1].scatter(Gb2[labels1_ == lab, 0], Gb2[labels1_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='r', marker=marker)
                for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
                    axes[1, 0].scatter(xb2[labels2_ == lab, 0], xb2[labels2_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='r', marker=marker)
                    axes[1, 1].scatter(Gb1[labels2_ == lab, 0], Gb1[labels2_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='b', marker=marker)
                fig.canvas.draw()
                plt.pause(1)

    # Change: Recording
    Gb1 = magan.get_layer(xb1_, xb2_, 'Gb1')
    Gb2 = magan.get_layer(xb1_, xb2_, 'Gb2')

    # Change: Reusability
    tf.reset_default_graph()

    return ([Gb1, Gb2], history, magan)
