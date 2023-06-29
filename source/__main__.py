# import welly
import lasio

# from entities import RandomForest

# well = welly.Well.from_las(


if __name__ == "__main__":
    la = lasio.read(
        r"C:\Users\Mfon\Desktop\Education\school stuffs\500"
        r"\Project\Loggy\data\spwla_volve\15_9-19A\15_9-19_A_CPI.las"
    )
    print(la)
