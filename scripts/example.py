"""Write a python function to remove all instances of the letter e from the string then
replace all spaces with exclamation points in the string then convert the string s to
lowercase then remove the first and last two characters of the string then remove every
third character from the string"""
def process(s):
    s =  s.replace("e", "")
    s =  s.replace(" ", "!")
    s =  s.lower()
    s =  s[2:-2]
    s =  "".join(char for i, char in enumerate(s) if i % 3 != 0)

assert (process('The quick brown fox jumps over a lazy dog!') ==
        'quckbrwnfo!jmp!or!!lzydo')
assert (process('How quickly daft jumping zebras vex!') == '!qiclydat!umin!zra!v')
assert (process('0123456789abcdefABCDEF') == '34679acdabd')
assert (process('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~') == "$%'(*+-.:;=>@[]^`{")
assert (process('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute
irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
deserunt mollit anim id est laborum?') ==
    'm!psm!olr!itam,!oncturadpicig!it!s!d!ismd!mpr!ncdiun!u!lbo!tdoormanaalqu.!t!imadmii
mvnam!qisnotrd!rctaio!ulacolaors!is!u!aiqipx!!cmmdocosqat!disau!iurdoorinrphnri!i!vluta!
vitsscilu!dlo!ufuia!nll!pritu.!xctu!sntocact!updaatno!point!sntincupaqu!ofiiadsun!mllt!n
i!i!s!lbou')

