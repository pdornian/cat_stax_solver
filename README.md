# cat_stax_solver
https://gamewright.com/product/cat-stax

A half assed, bug filled, non optimized, brute force solver in progress.

I started writing this because I got stuck on the three layer problems and thought it would be funny.

Used as a sandbox lab for me to teach myself things, mostly. I'm pretty sure it works on all 48 puzzles, if I coded them all. But I haven't.
Current worse case tested is puzzle 43, which took 12 minutes to solve. (Final puzzle 48 only took 2 minutes).

## Todo Wishlist (possibly never)
Most of these are just things on my list of coding skills I want to practice/learn.

- Solving routine is spaghetti and in need of refactoring
  * It's real silly. Not sure if it's salvagable.
- Algorithm optimization.
  * It's just brute force plus extremely minor heuristics right now.
  * Judging by google, some people have approached this by converting problem to SAT and then using solver for that. But I don't wanna do that because I like geometric intuition.
- Multiprocessing
- Proper error handling/runtime limits
- Implement all 48 puzzles
   * Figure out how they should be organized so loading them isn't an ``` import * ``` operation.
- Whatever else needs to be done to make it a real package that could be distributed
  * Not because it really should be one but because I've never done it and wanna learn.
- UI?
  * JK, probably never happening. I don't know how front ends work.
  * Might be useful for making process for creating new grids/puzzles because manually specifying forbidden slices by counting indices sucks.
