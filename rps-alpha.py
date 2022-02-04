#Neural network wich plays Rock, Paper, Scissors. Made by Jesus Daniel Diaz Barriga for the Visual Programming subject in Electronic Engineering 
#at the Veracruz Institute of Technology
from microMLP import MicroMLP
import time

mlp = MicroMLP.Create( neuronsByLayers           = [2, 2, 1],
                       activationFuncName        = MicroMLP.ACTFUNC_GAUSSIAN,
                       layersAutoConnectFunction = MicroMLP.LayersFullConnect )
                       
nnrock  = MicroMLP.NNValue.FromAnalogSignal(0)
nnpaper = MicroMLP.NNValue.FromAnalogSignal(1)
nnscissors= MicroMLP.NNValue.FromAnalogSignal(2)

mlp.AddExample( [nnrock, nnpaper], [nnpaper] )
mlp.AddExample( [nnrock, nnscissors], [nnrock] )
mlp.AddExample( [nnpaper , nnscissors ], [nnscissors] )
mlp.AddExample( [nnscissors, nnrock], [nnrock] )
mlp.AddExample( [nnscissors, nnpaper], [nnscissors] )
mlp.AddExample( [nnpaper, nnpaper], [nnpaper] )
mlp.AddExample( [nnrock, nnrock], [nnrock] )
mlp.AddExample( [nnscissors, nnscissors], [nnscissors] )


learnCount = mlp.LearnExamples()
 

 
# Set of instructions for Rock-Paper-Scissors
def rps_instructions():
 
    print()
    print("Instructions for Rock-Paper-Scissors : ")
    print()
    print("Rock crushes Scissors")
    print("Scissors cuts Paper")
    print("Paper covers Rock")
    print()

def rps():
     
    global name
 
    # Game Loop for each game of Rock-Paper-Scissors
    while True:
 
        print("--------------------------------------")
        print("\t\tMenu")
        print("--------------------------------------")
        print("Enter \"help\" for instructions")
        print("Enter \"Rock\",\"Paper\",\"Scissors\" to play")
        print("Enter \"exit\" to quit")
        print("--------------------------------------")
 
        print()
 
        # Player Input
        inp = input("Enter your move : ")
 
        if inp.lower() == "help":
            
            rps_instructions()
            continue
        elif inp.lower() == "exit":
            
            break  
        elif inp.lower() == "rock":
            player_move = 0
        elif inp.lower() == "paper":
            player_move = 1    
        elif inp.lower() == "scissors":
            player_move = 2
        else:
            
            print("Wrong Input!!")
            rps_instructions()  
            continue
 
        print("Computer making a move....")
 
        print()
        time.sleep(2)
        #Declare what neural network chose to get "1" from playing with itself
        #if (player 1 move - player 2 move)%3 = 1, player 1 wins
        #Based on player input we ask for the outcome when nn wins only if it learned well
        if (player_move ==0):
            nn=(mlp.Predict([nnrock, nnpaper])[0].AsInt)
            if(nn==1):
                nn_move="paper"
                nn_value=1
            else:
                nn_move="rock"
                nn_value=0
        elif (player_move==1):
            nn=(mlp.Predict([nnscissors , nnpaper] )[0].AsInt)
            if(nn==1):
                nn_move="scissors"
                nn_value=2
            else:
                nn_move="paper"
                nn_value=1
        elif (player_move==2):
            nn=(mlp.Predict([nnrock, nnscissors] )[0].AsInt)
            if(nn==1):
                nn_move="rock"
                nn_value=0
            else:
                nn_move="scissors"
                nn_value=2
        # Get the learned computer move 
        comp_move = nn
 
        # Print the computer move
        print("Computer chooses ", nn_move)
 
        # Find the winner of the match
        winner=(nn_value - player_move)%3
                
        # Declare the winner 
        if winner == 1:
            print("COMPUTER WINS!!!")
        elif (player_move==nn_value):
            print("TIE GAME")
        else:
            print(name, "WINS!!!")
        print()
        time.sleep(2)
        
 
def rpsls():
     
    global name
 
 
# The main function
if __name__ == '__main__':
 
 
    print( "LEARNED :" )
    print( "  - Rock v Paper = %s" % mlp.Predict([nnrock, nnpaper])[0].AsInt  )
    print( "  - Rock v Scissors  = %s" % mlp.Predict([nnrock, nnscissors] )[0].AsInt  )
    print( "  - Scissors v Paper  = %s" % mlp.Predict([nnscissors , nnpaper] )[0].AsInt  )
 
    name = input("Enter your name: ")
  
    # The GAME LOOP
    while True:
 
        # The Game Menu
        print()
        print("Let's Play!!!")
        
        print("Enter 1 to play Rock-Paper-Scissors")
        
        print("Enter 3 to quit")
        print()
 
        # Try block to handle the player choice 
        try:
            choice = int(input("Enter your choice = "))
        except ValueError:
            
            print("Wrong Choice")   
            continue
 
        # Play the traditional version of the game
        if choice == 1:
            rps()
  
        # Quit the GAME LOOP    
        elif choice == 3:
            break
 
        # Other wrong input
        else:
            
            print("Wrong choice. Read instructions carefully.")
