import CG.train as CGT 

def main():
    CGT.main(["--dataroot", "./catdog",
              "--n_epochs", "20"])

if __name__ == "__main__":
    main()