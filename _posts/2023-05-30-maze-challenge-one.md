---
title: challenge 1. May the Maze Be with You
published: true
description: Early into the challenge we adapted a DFS algorithm to find a solution path through the maze. The maze, modeled as a graph, was quickly solvable using this approach.
category: writeup
author: gary23w
featured_image: /assets/images/post2.png
category: 
    - CTF
tags:
    - CTF
    - defcon
    - challenge1
---

# Maze Challenge Analysis and Solution

# Binary breakdown

- [./challenge](#challenge)
  - [main](#main)
  - [generate_maze](#generate_maze)
  - [rand_range](#rand_range)
  - [randomDescription](#randomDescription)
  - [validWalk](#validWalk)
  - [winner](#winner)

## Analysis

Our journey in solving this maze challenge started with a collaborative exploration and bifurcated strategies. Brad and I, each having a unique approach to the problem, decided to investigate the maze independently. Brad focused on a depth-first search (DFS) approach, while I opted to explore the breadth-first search (BFS) methodology.

DFS and BFS are well-known algorithms used to traverse or search graphs or tree data structures. The choice between the two primarily depends on the nature of the problem and specific constraints at hand.

Early into the challenge, we adapted a DFS algorithm to find a solution path through the maze. The maze, modeled as a graph, was quickly solvable using this approach. The adaptability of DFS was indeed instrumental during this phase. However, we soon ran into a more complex obstacle that required a different set of tools: timing.

The challenge had a built-in timing mechanism that added a layer of complexity to our solution. Our initial efforts to circumvent this timing issue resulted in two different brute force versions of our script. Unfortunately, despite our rigorous trials, we were unable to conclusively solve the problem using these scripts.

Starting late on this challenge, we found ourselves up against the clock, with only two hours to solve it. At this point, we decided to step back, replenish ourselves, and attack the problem with fresh minds on another day.

Two days later, we reconvened and developed a strategy that utilized ctypes, a powerful Python library that allows Python code to call C functions in dynamic link libraries/shared libraries. Our strategy was to use ctypes to dynamically load the libc.so.6 library, a commonly used library that includes crucial C standard functions.

Specifically, we utilized ctypes to access the rand and srand functions from the libc.so.6 library. The srand function is used to initialize the random number generator in C, and rand is used to generate random numbers. By seeding the random number generator with a predictable value (the current time), we were able to control the sequence of random numbers produced by rand.

This strategy finally led us to the crux of our solution: a successful run of the script that navigates the maze, satisfies the timing constraints, and handles random seeding effectively, solving the maze challenge in its entirety. Our journey highlights the necessity of adaptive problem-solving strategies, collaboration, and resilience in the face of challenging obstacles.

## Detailed Breakdown of the Code

The solution was structured based on the Observer pattern using two major components - `ForceSensitive` and `MidiChlorian`.

### MidiChlorian Class

This class is designed as the Observable component of our Observer design pattern. It contains a list of observers, referred to as `jedis`. It also contains methods to add or remove observers to this list (`recruit_jedi` and `expel_jedi`). The `trigger_force` method is the most critical part of this class. When called, it notifies all observers about an event, passing them the event's arguments.

### MazeRunner Class

Inheriting from the `MidiChlorian` class, the `MazeRunner` class represents the maze solver part of our solution. In the constructor (`__init__` method), it takes the maze and the start and end points as parameters. The `escape_route` method initiates the solving process. It first creates a two-dimensional array `rebels` that tracks visited cells, and then calls the `kenobi_guidance` method to find a path. If a path is found, it triggers the force to notify all observers.

The `kenobi_guidance` method performs a depth-first search to find a path from the start point to the end point. It uses recursion to try out all possible paths. If it reaches the end point, it returns a list containing the directions to the end point. If it reaches a cell that has already been visited or is a wall, it returns False, signifying that no path can be found in this direction.

## DeathStarChallengeHandler Class

The DeathStarChallengeHandler class plays a crucial role in this solution, serving as the intermediary between the challenge server and our maze solver (MazeRunner). It's equipped to handle the challenge's binary, communication with the server, and dynamic linking of shared libraries via ctypes.

#### Constructor

The constructor of this class begins by loading two ELF (Executable and Linkable Format) binaries using the pwnlib.ELF class: the challenge binary and the ld-linux-x86-64.so.2 binary. ELF is a common standard file format for executables, object code, shared libraries, and core dumps. Loading these binaries allows the script to interact with them directly.

The constructor also sets up multiple communication channels (obi_wan_comm, anakin_comm, yoda_comm, and leia_comm) with the server using different functions from the pwnlib. These functions are designed to send or receive data until a certain condition is met, providing a high level of control over the interaction with the challenge server.

the constructor uses ctypes, a Python library for calling C functions in dynamic link libraries/shared libraries, and for creating, accessing and manipulating C data types in Python. It loads the libc.so.6 library and seeds its random number generator with the current time. This random number generator is used later in the han_solo method to generate random numbers in a predictable manner.

depending on whether the script is run in a debugging environment or not, the constructor sets up the connection to the challenge server accordingly. It can either start a remote connection to the challenge server, start the challenge binary in a local debugging process, or start it as a normal local process.

#### star_wars Method

The star_wars method serves as the main driver for the challenge solution. It first invokes the death_star_challenge method, which establishes the communication with the challenge server and retrieves the maze as well as the starting and ending points. Then, a MazeRunner object is created with these parameters and the DeathStarChallengeHandler instance is registered as an observer to this object. Finally, the escape_route method is called on the MazeRunner object, initiating the maze-solving process.

#### perceive Method

This method acts as the event handler, triggered whenever the MazeRunner instance (the observable) finds a path through the maze. Upon invocation, this method receives the identified path as an argument and passes it on to the han_solo method for further processing.

#### han_solo Method

The han_solo method is responsible for communicating the identified path back to the challenge server. It does this by iterating over the path and for each direction, generating random numbers until it gets a number that modulo 1213 equals 1212. This peculiar operation is based on the requirements of the challenge and after each successful operation, it sends the corresponding direction back to the server. Once it has sent all directions, it waits for the challenge to end, signifying a successful completion.

Through the careful orchestration of ctypes, dynamic libraries, and inter-process communication, the DeathStarChallengeHandler class forms the cornerstone of our solution to the maze challenge.

## Conclusion

With careful implementation of the Observer design pattern, usage of depth-first search for maze navigation, and clever handling of communication with the challenge server, we were able to tackle this challenging maze problem effectively.

# Solution:

{% raw %}

```python
from abc import ABC, abstractmethod
from pwn import \*
import ctypes

class ForceSensitive(ABC):
@abstractmethod
def perceive(self, midi_chlorian, \*args):
pass

class MidiChlorian:
def **init**(self):
self.jedis = []

    def recruit_jedi(self, jedi):
        self.jedis.append(jedi)

    def expel_jedi(self, jedi):
        self.jedis.remove(jedi)

    def trigger_force(self, *args):
        for jedi in self.jedis:
            jedi.perceive(self, *args)

class MazeRunner(MidiChlorian):
def **init**(self, death_star, luke, leia, vader, palpatine):
super().**init**()
self.death_star = death_star
self.luke = luke
self.leia = leia
self.vader = vader
self.palpatine = palpatine

    def escape_route(self):
        rebels = [[False for _ in range(len(self.death_star[0]))] for _ in range(len(self.death_star))]
        path = self.kenobi_guidance(self.luke, self.leia, rebels)
        self.trigger_force(path)

    def kenobi_guidance(self, yoda, mace, rebels):
        if rebels[yoda][mace]:
            return False
        rebels[yoda][mace] = True

        for i, (dy, dx) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            jedi, padawan = yoda + dy, mace + dx
            if jedi < 0 or jedi >= len(self.death_star) or padawan < 0 or padawan >= len(self.death_star[0]):
                continue
            if self.death_star[jedi][padawan] == '#':
                continue
            if jedi == self.vader and padawan == self.palpatine:
                return [i]
            if res := self.kenobi_guidance(jedi, padawan, rebels):
                return [i] + res
        return False

class DeathStarChallengeHandler(ForceSensitive):
def **init**(self):
self.defcon1 = "challenge"
self.death_star = ELF(self.defcon1, checksec=True)
self.force = ELF("ld-linux-x86-64.so.2", checksec=True)
context.binary = self.death_star
self.obi_wan_comm = lambda *x: self.r2d2.recvuntil(*x)
self.anakin_comm = lambda *x: self.r2d2.recvline(*x)
self.yoda_comm = lambda *x: self.r2d2.sendlineafter(*x)
self.leia_comm = lambda *x: self.r2d2.sendline(*x)
self.force_chlorian = ctypes.CDLL("libc.so.6")
self.force_chlorian.srand(self.force_chlorian.time(0))

        self.var = os.getenv("ILIKEBIGBOOBIES")
        if self.var is None:
            self.HOST = os.environ.get("HOST", "localhost")
            self.PORT = 31337
            self.r2d2 = remote(self.HOST, int(self.PORT))
        elif args.GDB:
            self.r2d2 = gdb.debug(f"darkside/{self.defcon1}", "c", aslr=False)
        else:
            self.r2d2 = process(f"darkside/{self.defcon1}")

    def perceive(self, midi_chlorian, *args):
        path = args[0]
        self.han_solo(path)

    def star_wars(self):
        death_star, luke, leia, vader, palpatine = self.death_star_challenge()
        runner = MazeRunner(death_star, luke, leia, vader, palpatine)
        runner.recruit_jedi(self)
        runner.escape_route()

    def death_star_challenge(self):
        self.obi_wan_comm(b"Welcome to the maze!\n")
        self.leia_comm(b"a")
        self.obi_wan_comm(b"end the torment: ")
        death_star = [self.anakin_comm().strip().decode() for _ in range(29)]

        luke, leia, vader, palpatine = -1, -1, -1, -1
        n = len(death_star)
        m = len(death_star[0])

        for i in range(n):
            for j in range(m):
                if death_star[i][j] == '@':
                    luke, leia = i, j
                elif death_star[i][j] == '*':
                    vader, palpatine = i, j
        return death_star, luke, leia, vader, palpatine

    def han_solo(self, path):
        for _ in range(591):
            self.force_chlorian.rand()

        direction = {0: 's', 1: 'n', 2: 'e', 3: 'w'}
        for i in path[:-1]:
            self.force_chlorian.rand()
            self.yoda_comm(b"end the torment: ", direction[i].encode())
            luke, leia = luke + [1, -1, 0, 0][i], leia + [0, 0, 1, -1][i]

        for i, (dy, dx) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            jedi, padawan = luke + dy, leia + dx
            if death_star[jedi][padawan] == '#':
                ddir = i
                break

        while self.force_chlorian.rand() % 1213 != 1212:
            self.yoda_comm(b"end the torment: ", direction[ddir].encode())

        self.yoda_comm(b"end the torment: ", direction[path[-1]].encode())

        self.r2d2.recvall(timeout=1)

handler = DeathStarChallengeHandler()
handler.star_wars()
```

{% endraw %}

---

<a name="challenge"></a>

## ./challenge

<a name="main"></a>

### main:

{% raw %}

```cpp
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define MAZE_SIZE 30

int main(void)
{
int playerX, playerY, show_maze = 0;
char player_move;
char maze[MAZE_SIZE][MAZE_SIZE];
long stack_chk_guard;

    stack_chk_guard = stack_chk_guard_value; // Save the stack guard value
    srand(time(NULL)); // Seed the random number generator

    // Generate the maze with walls
    for (int i = 0; i < MAZE_SIZE; i++) {
        for (int j = 0; j < MAZE_SIZE; j++) {
            maze[i][j] = '#';
        }
    }

    puts("Reticulating splines...");
    generate_maze(maze, 1, 1, '\x01');

    puts("\n\nWelcome to the maze!");

    // Initialize player position
    playerX = 1;
    playerY = 1;

    do {
        if (show_maze != 0) {
            display_maze(maze, playerX, playerY);
        }

        printf("You are in room (%d, %d)\n", playerX, playerY);

        // Print possible moves
        print_options(maze, playerX, playerY);

        scanf(" %c", &player_move);
        putchar('\n');

        // Handle player movement or special actions
        switch(player_move) {
            case 'a':
                puts("You cast arcane eye and send your summoned magical eye above the maze.");
                show_maze = 1;
                break;
            case 'e':
                if (valid_move(maze, playerX, playerY + 1)) {
                    playerY++;
                }
                break;
            case 'n':
                if (valid_move(maze, playerX - 1, playerY)) {
                    playerX--;
                }
                break;
            case 'q':
                puts("Ending the game.");
                exit(0);
                break;
            case 's':
                if (valid_move(maze, playerX + 1, playerY)) {
                    playerX++;
                }
                break;
            case 'w':
                if (valid_move(maze, playerX, playerY - 1)) {
                    playerY--;
                }
                break;
        }
    } while (maze[playerX][playerY] != '*');

    if (rand() % 0x4bd == 0x4bc) {
        puts("You successfully exit the maze!");
        winner();
        return 0;
    }

    puts("Just as you are about to exit, a displacer beast captures you. You die.");
    exit(0);

}
```

{% endraw %}
<a name="generate_maze"></a>

### generate_maze

{% raw %}

```cpp
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void generate_maze(char \*maze, int startX, int startY, char initialCharacter)
{
int directionArr[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
int swapIdx, swapTemp[2];
int newPosX, newPosY;

    // Set the starting position in the maze to '.' (empty space)
    maze[startX * MAZE_SIZE + startY] = '.';

    // Shuffle the direction array
    for (int i = 3; i > 0; i--) {
        swapIdx = rand_range(0, i);

        // Swap directionArr[i] and directionArr[swapIdx]
        swapTemp[0] = directionArr[i][0];
        swapTemp[1] = directionArr[i][1];

        directionArr[i][0] = directionArr[swapIdx][0];
        directionArr[i][1] = directionArr[swapIdx][1];

        directionArr[swapIdx][0] = swapTemp[0];
        directionArr[swapIdx][1] = swapTemp[1];
    }

    // Try to extend the maze in each direction
    for (int i = 0; i < 4; i++) {
        newPosX = startX + directionArr[i][0] * 2;
        newPosY = startY + directionArr[i][1] * 2;

        // If the new position is within bounds and has not been visited
        if (newPosX > 0 && newPosX < MAZE_SIZE - 1 && newPosY > 0 && newPosY < MAZE_SIZE - 1
            && maze[newPosX * MAZE_SIZE + newPosY] == '#') {

            // Set the new position and the wall between the current and new position to '.'
            maze[startX * MAZE_SIZE + startY] = '.';
            generate_maze(maze, newPosX, newPosY, 0);
        }
    }

    // If initialCharacter is not 0, this is the first call to generate_maze
    if (initialCharacter != 0) {
        int endPosX, endPosY;

        if (rand_range(0, 1) == 0) {
            do {
                endPosX = rand_range(1, MAZE_SIZE - 2);
            } while (maze[endPosX * MAZE_SIZE + MAZE_SIZE - 2] != '.');

            // Mark the exit point with '*'
            maze[endPosX * MAZE_SIZE + MAZE_SIZE - 1] = '*';
        }
        else {
            do {
                endPosY = rand_range(1, MAZE_SIZE - 2);
            } while (maze[(MAZE_SIZE - 2) * MAZE_SIZE + endPosY] != '.');

            // Mark the exit point with '*'
            maze[(MAZE_SIZE - 1) * MAZE_SIZE + endPosY] = '*';
        }

        // Clear the last row and column
        for (int i = 0; i < MAZE_SIZE; i++) {
            maze[(MAZE_SIZE - 1) * MAZE_SIZE + i] = ' ';
            maze[i * MAZE_SIZE + MAZE_SIZE - 1] = ' ';
        }
    }

}
```

{% endraw %}
<a name="rand_range"></a>

### rand_range

{% raw %}

```cpp
char spinner;
int spinner_counter;
int random_counter;

int rand_range(int lower, int upper)
{
// Print a spinner character
printf("\r%c", spinner + spinner_counter);

    // Cycle spinner counter
    int spinnerLength = strlen(&spinner);
    spinner_counter = (spinner_counter + 1) % spinnerLength;

    // Flush the output buffer and introduce a slight delay
    fflush(stdout);
    usleep(5000);

    // Increment the random counter
    random_counter++;

    // Generate a random number in the range [lower, upper]
    int randomValue = rand();
    return lower + randomValue % (upper - lower + 1);

}
```

{% endraw %}
<a name="randomDescription"></a>

### randomDescription

{% raw %}

```cpp
#include <stdio.h>
#include <stdlib.h>

// Assuming the following is a global variable
int random_counter;

void randomDescription()
{
long in_FS_OFFSET;
char *roomSizes[] = {"cozy", "medium-sized", "spacious", "massive"};
char *items[] = {"bookshelves", "fireplaces", "suits of armor", "tables", "chests",
"paintings", "statues", "tapestries", "candelabras", "chairs",
"fountains", "mirrors", "curtains", "chess sets"};
char \*styles[] = {"Art Deco", "Baroque", "Classical", "Colonial", "Contemporary", "Country",
"Gothic", "Industrial", "Mediterranean", "Minimalist", "Neoclassical",
"Renaissance", "Rococo", "Romantic", "Rustic", "Victorian"};
long local_10;

    local_10 = *(long *)(in_FS_OFFSET + 0x28);

    // Increment the random counter
    random_counter++;

    // Generate a random number
    uint randomNum = rand();

    // Select random elements from the arrays using the generated random number
    char* randomRoomSize = roomSizes[randomNum & 3];
    char* randomItem1 = items[randomNum >> 2 & 0xF];
    char* randomItem2 = items[randomNum >> 6 & 0xF];
    int randomFlowersCount = randomNum >> 0xE & 0x1F;
    int randomStarsCount = randomNum >> 0xE;
    char* randomStyle = styles[randomNum >> 10 & 0xF];

    // Print the generated description
    printf("As you step into the room, you find yourself standing in a %s space. The walls are adorned with %s and two large %s dominate the center of the room. You see %d flowers in a vase, and through a window you stop to count %d stars. The room appears well designed in the %s style.\n",
           randomRoomSize, randomItem1, randomItem2, randomFlowersCount, randomStarsCount, randomStyle);

    // Stack smashing protection
    if (local_10 != *(long *)(in_FS_OFFSET + 0x28)) {
        // If stack corruption detected, do not return from function
        __stack_chk_fail();
    }

    return;

}
```

{% endraw %}
<a name="validWalk"></a>

### validWalk

{% raw %}

```cpp
bool validwalk(char direction) {
// The function checks if the provided direction character is within the ASCII range of 39 (')') to 47 ('/')
if (direction < '\'' || direction > '/') {
return false;
}
else {
return true;
}
}
```

{% endraw %}
<a name="winner"></a>

### winner

{% raw %}

```cpp
#include <stdlib.h>

void winner() {
// Print the victory message
printf("Congratulations! You have solved the maze!\n");

    // Open a shell
    system("/bin/sh");

    // Exit the program
    exit(0);

}
```

{% endraw %}
