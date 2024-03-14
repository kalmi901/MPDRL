from envs import Pos1B1D



if __name__ == "__main__":
    venc = Pos1B1D(16)

    venc.reset()

    for ic in range(10):

        venc.step()
        venc.render()

        input()

