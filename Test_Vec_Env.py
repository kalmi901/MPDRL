from envs import Pos1B1D



if __name__ == "__main__":
    venc = Pos1B1D(1024)

    venc.reset()

    for ic in range(500):

        venc.step(venc.action_space.sample())
        venc.render()

       
