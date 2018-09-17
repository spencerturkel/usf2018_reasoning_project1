"""This is basically the file I will use to grade your HW1 submissions.
Obviously, the test problems hard-coded in here will be different. You
can use this to test your code. I'm using Python 3.4+.
- Dr. Licato
"""

import random
import traceback
import time


studentName = "TestStudent"
#there will likely be 5-10 problems
problems = ['p',
	'(NOT (NOT (NOT (NOT not))  )		)',
	'(IF p p)',
	'(AND p (NOT p))'] 
answers = [1,
	1,
	'T',
	'U']
	
maxProblemTimeout = 30


outFile = open("grade_"+studentName+".txt", 'w')

def prnt(S):
	global outFile
	outFile.write(str(S) + "\n")
	print(S)

try:
	F = open("p1.py", 'r')
	exec("".join(F.readlines()))
except Exception as e:
	prnt("Couldn't open or execute 'hw1.py': " + str(traceback.format_exc()))
	prnt("FINAL SCORE: 0")
	exit()


currentScore = 100
for i in range(len(problems)):
	P = problems[i]
	A = answers[i]
	
	prnt('='*30)
	prnt("TESTING ON INPUT PROBLEM:")
	prnt(P)
	prnt("CORRECT OUTPUT:")
	prnt(str(A))
	prnt("YOUR OUTPUT:")
	try:
		startTime = time.time()
		result = proveFormula(P)
		prnt(result)
		endTime = time.time()		
		if endTime-startTime > maxProblemTimeout:
			prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (-10 points)")
		elif result==A:
			prnt("Correct!")
		else:
			prnt("Incorrect (-10 points)")
			currentScore -= 10

	except Exception as e:
		prnt("Error while executing this problem: " + str(traceback.format_exc()))
		currentScore -= 10
	
prnt('='*30)
prnt('='*30)
prnt('='*30)
prnt("FINAL SCORE:" + str(currentScore))