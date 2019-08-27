from tqdm import trange


def train(self, model=None, x=None, optimizer=None, sequence_length=None, loss=None, n_epochs=None, criterion=None):

    error = 0

    # epoch loop
    for e in trange(n_epochs):

        # sample loop
        for i in range(x.shape[1]):

            predicted = model(input=x[:, i])

            error = error + loss(predicted, observed=x[:, i])

            # truncation
            if (i > 0) & ((i % sequence_length) == 0):

                # compute gradients
                error.backward()

                # update parameters
                optimizer.step()

                model.zero_grad()          # clear the gradients
                predicted.detach_()        # detach output

                # reinitialize the hidden states

                model.detach_states()  # detach hidden states
                model.init_hidden(shape=(x.shape[0], model.hidden_dim))

            # set error back to zero
            error = 0
