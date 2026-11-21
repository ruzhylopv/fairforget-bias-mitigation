# Tokenization check: example pairs

## lapa-12b-instruct

### stereo multi-token (1061 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | drunks | athletic | `▁dr|unks` | `▁athletic` | word_start / word_start |
| profession | stingy | generous | `▁sting|y` | `▁generous` | word_start / word_start |
| gender | gossiping | coding | `▁goss|iping` | `▁coding` | word_start / word_start |
| religion | isis | peace | `▁is|is` | `▁peace` | word_start / word_start |
| race | sexing | teaching | `▁sex|ing` | `▁teaching` | word_start / word_start |

### anti multi-token (932 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `▁dishonest` | `▁chival|rous` | word_start / word_start |
| gender | ceo | homemaker | `▁ceo` | `▁homem|aker` | word_start / word_start |
| profession | daydreaming | weightlifting | `▁daydream|ing` | `▁weight|lifting` | word_start / word_start |
| religion | terrorism | peacefulness | `▁terrorism` | `▁peaceful|ness` | word_start / word_start |
| race | terrorists | pacifists | `▁terrorists` | `▁pacif|ists` | word_start / word_start |

### stereo fewer tokens than anti (789 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `▁dishonest` | `▁chival|rous` | word_start / word_start |
| gender | ceo | homemaker | `▁ceo` | `▁homem|aker` | word_start / word_start |
| profession | creative | unimaginative | `▁creative` | `▁unimagin|ative` | word_start / word_start |
| religion | terrorism | peacefulness | `▁terrorism` | `▁peaceful|ness` | word_start / word_start |
| race | terrorists | pacifists | `▁terrorists` | `▁pacif|ists` | word_start / word_start |

### stereo more tokens than anti (902 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | drunks | athletic | `▁dr|unks` | `▁athletic` | word_start / word_start |
| profession | stingy | generous | `▁sting|y` | `▁generous` | word_start / word_start |
| gender | gossiping | coding | `▁goss|iping` | `▁coding` | word_start / word_start |
| religion | isis | peace | `▁is|is` | `▁peace` | word_start / word_start |
| race | sexing | teaching | `▁sex|ing` | `▁teaching` | word_start / word_start |

### first difference mid-word (0 / 8498 pairs)

(none)

### first difference outside the word (0 / 8498 pairs)

(none)

## gemma3-12b-it

### stereo multi-token (1061 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | drunks | athletic | `▁dr|unks` | `▁athletic` | word_start / word_start |
| profession | stingy | generous | `▁sting|y` | `▁generous` | word_start / word_start |
| gender | gossiping | coding | `▁goss|iping` | `▁coding` | word_start / word_start |
| religion | isis | peace | `▁is|is` | `▁peace` | word_start / word_start |
| race | sexing | teaching | `▁sex|ing` | `▁teaching` | word_start / word_start |

### anti multi-token (932 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `▁dishonest` | `▁chival|rous` | word_start / word_start |
| gender | ceo | homemaker | `▁ceo` | `▁homem|aker` | word_start / word_start |
| profession | daydreaming | weightlifting | `▁daydream|ing` | `▁weight|lifting` | word_start / word_start |
| religion | terrorism | peacefulness | `▁terrorism` | `▁peaceful|ness` | word_start / word_start |
| race | terrorists | pacifists | `▁terrorists` | `▁pacif|ists` | word_start / word_start |

### stereo fewer tokens than anti (789 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `▁dishonest` | `▁chival|rous` | word_start / word_start |
| gender | ceo | homemaker | `▁ceo` | `▁homem|aker` | word_start / word_start |
| profession | creative | unimaginative | `▁creative` | `▁unimagin|ative` | word_start / word_start |
| religion | terrorism | peacefulness | `▁terrorism` | `▁peaceful|ness` | word_start / word_start |
| race | terrorists | pacifists | `▁terrorists` | `▁pacif|ists` | word_start / word_start |

### stereo more tokens than anti (902 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | drunks | athletic | `▁dr|unks` | `▁athletic` | word_start / word_start |
| profession | stingy | generous | `▁sting|y` | `▁generous` | word_start / word_start |
| gender | gossiping | coding | `▁goss|iping` | `▁coding` | word_start / word_start |
| religion | isis | peace | `▁is|is` | `▁peace` | word_start / word_start |
| race | sexing | teaching | `▁sex|ing` | `▁teaching` | word_start / word_start |

### first difference mid-word (0 / 8498 pairs)

(none)

### first difference outside the word (0 / 8498 pairs)

(none)

## qwen3-8b

### stereo multi-token (1686 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| profession | nerdy | lazy | `Ġner|dy` | `Ġlazy` | word_start / word_start |
| race | drunks | athletic | `Ġdr|unks` | `Ġathletic` | word_start / word_start |
| gender | ceo | homemaker | `Ġce|o` | `Ġhom|emaker` | word_start / word_start |
| religion | intolerant | tolerant | `Ġintoler|ant` | `Ġtolerant` | word_start / word_start |
| profession | stingy | generous | `Ġsting|y` | `Ġgenerous` | word_start / word_start |

### anti multi-token (1480 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `Ġdishonest` | `Ġch|ival|rous` | word_start / word_start |
| profession | romantic | illiterate | `Ġromantic` | `Ġill|iterate` | word_start / word_start |
| gender | ceo | homemaker | `Ġce|o` | `Ġhom|emaker` | word_start / word_start |
| religion | terrorism | peacefulness | `Ġterrorism` | `Ġpeaceful|ness` | word_start / word_start |
| race | smelly | fragrant | `Ġsm|elly` | `Ġfrag|rant` | word_start / word_start |

### stereo fewer tokens than anti (1173 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `Ġdishonest` | `Ġch|ival|rous` | word_start / word_start |
| profession | romantic | illiterate | `Ġromantic` | `Ġill|iterate` | word_start / word_start |
| gender | social | antisocial | `Ġsocial` | `Ġantis|ocial` | word_start / word_start |
| religion | terrorism | peacefulness | `Ġterrorism` | `Ġpeaceful|ness` | word_start / word_start |
| race | terrorists | pacifists | `Ġterrorists` | `Ġpac|if|ists` | word_start / word_start |

### stereo more tokens than anti (1339 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| profession | nerdy | lazy | `Ġner|dy` | `Ġlazy` | word_start / word_start |
| race | drunks | athletic | `Ġdr|unks` | `Ġathletic` | word_start / word_start |
| gender | macho | delicate | `Ġmach|o` | `Ġdelicate` | word_start / word_start |
| religion | intolerant | tolerant | `Ġintoler|ant` | `Ġtolerant` | word_start / word_start |
| profession | stingy | generous | `Ġsting|y` | `Ġgenerous` | word_start / word_start |

### first difference mid-word (3 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | iraq | iran | `Ġir|aq` | `Ġir|an` | mid_word / mid_word |
| profession | arent | are | `Ġare|nt` | `Ġare` | mid_word / after_word |
| race | oman | omani | `Ġo|man` | `Ġo|mani` | mid_word / mid_word |

### first difference outside the word (1 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | rome | mexico | `rome` | `Ġmexico` | before_word / word_start |

## llama31-8b-instruct

### stereo multi-token (1685 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| profession | nerdy | lazy | `Ġner|dy` | `Ġlazy` | word_start / word_start |
| race | drunks | athletic | `Ġdr|unks` | `Ġathletic` | word_start / word_start |
| gender | ceo | homemaker | `Ġce|o` | `Ġhom|emaker` | word_start / word_start |
| religion | intolerant | tolerant | `Ġintoler|ant` | `Ġtolerant` | word_start / word_start |
| profession | stingy | generous | `Ġsting|y` | `Ġgenerous` | word_start / word_start |

### anti multi-token (1478 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `Ġdishonest` | `Ġch|ival|rous` | word_start / word_start |
| profession | romantic | illiterate | `Ġromantic` | `Ġill|iterate` | word_start / word_start |
| gender | ceo | homemaker | `Ġce|o` | `Ġhom|emaker` | word_start / word_start |
| religion | terrorism | peacefulness | `Ġterrorism` | `Ġpeaceful|ness` | word_start / word_start |
| race | smelly | fragrant | `Ġsm|elly` | `Ġfrag|rant` | word_start / word_start |

### stereo fewer tokens than anti (1172 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | dishonest | chivalrous | `Ġdishonest` | `Ġch|ival|rous` | word_start / word_start |
| profession | romantic | illiterate | `Ġromantic` | `Ġill|iterate` | word_start / word_start |
| gender | social | antisocial | `Ġsocial` | `Ġantis|ocial` | word_start / word_start |
| religion | terrorism | peacefulness | `Ġterrorism` | `Ġpeaceful|ness` | word_start / word_start |
| race | terrorists | pacifists | `Ġterrorists` | `Ġpac|if|ists` | word_start / word_start |

### stereo more tokens than anti (1338 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| profession | nerdy | lazy | `Ġner|dy` | `Ġlazy` | word_start / word_start |
| race | drunks | athletic | `Ġdr|unks` | `Ġathletic` | word_start / word_start |
| gender | macho | delicate | `Ġmach|o` | `Ġdelicate` | word_start / word_start |
| religion | intolerant | tolerant | `Ġintoler|ant` | `Ġtolerant` | word_start / word_start |
| profession | stingy | generous | `Ġsting|y` | `Ġgenerous` | word_start / word_start |

### first difference mid-word (3 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | iraq | iran | `Ġir|aq` | `Ġir|an` | mid_word / mid_word |
| profession | arent | are | `Ġare|nt` | `Ġare` | mid_word / after_word |
| race | oman | omani | `Ġo|man` | `Ġo|mani` | mid_word / mid_word |

### first difference outside the word (1 / 8498 pairs)

| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |
|---|---|---|---|---|---|
| race | rome | mexico | `rome` | `Ġmexico` | before_word / word_start |
