"""Hand-written hard paraphrases (2 per topic): same request as the training phrasings of the topic, but the topic noun
and the "Which X should I ...?" template are not reused. Written and screened ONLY by frozen-BERT cosine to the topic's
training phrasings (< 0.90) and word overlap, before any model was evaluated on them.

HARD_DRAFT1 (question form, first attempt) had only 2 of 24 sentences below 0.90, so HARD was rewritten as polite
statements ("Please indicate ...", "Kindly specify ..."); all 24 are below 0.90. The price: HARD differs from the
training phrasings in sentence form as well as in vocabulary."""
HARD = {
    "fruit":    ["Tell me what sort of produce you want cleaned and sliced.",
                 "I would like to hear your preference among the juicy things from the market stall."],
    "plate":    ["Please indicate the flat serving ware on which the pastries should be arranged.",
                 "Kindly specify the tableware for presenting your pasta."],
    "knife":    ["Kindly indicate the blade to use when chopping the greens.",
                 "It would help to know the sharp utensil you prefer for the vegetables."],
    "cup":      ["Kindly specify the drinkware for your smoothie.",
                 "Let me know which beverage container to take from the shelf."],
    "bowl":     ["Kindly specify the container for serving your rice.",
                 "Let me know which round vessel to fill with the porridge."],
    "yogurt":   ["Kindly specify the tangy dairy item you want served.",
                 "Let me know which fermented milk product to take from the fridge."],
    "bread":    ["Please indicate the sort of loaf that should be crisped up for your first meal.",
                 "Kindly specify the baked goods you want warmed."],
    "eggs":     ["Please indicate the cooking method you want for the hen's produce this morning.",
                 "Kindly specify the number to crack into the omelette pan."],
    "coffee":   ["Please indicate the roast of beans that should be ground for you.",
                 "Kindly specify the caffeinated drink you want made."],
    "tea":      ["Please indicate the herbal infusion that should be steeped for you.",
                 "Let me know which blend of dried herbs to pour hot water over."],
    "snack":    ["Please indicate the light bites that should be laid out for your guests.",
                 "Kindly specify the crunchy finger food you wish to have on hand."],
    "cleaning": ["Kindly specify the scrubbing product to use on the sink.",
                 "I would appreciate knowing what soap you want for the washing up."],
}

HARD_DRAFT1 = {
    "fruit":    ["What fresh produce do you fancy having today?",
                 "Of the sweet things that grow on trees, which one are you in the mood for?"],
    "plate":    ["What tableware do you want your lunch served on?",
                 "On what kind of dish would you like the meal presented?"],
    "knife":    ["What blade would suit the carving best?",
                 "Tell me which cutting tool you would rather I grab."],
    "cup":      ["In which mug would you like your drink poured?",
                 "What drinkware should I fetch for your beverage?"],
    "bowl":     ["What deep container shall I serve the stew in?",
                 "Into what vessel do you want me to ladle your broth?"],
    "yogurt":   ["What kind of cultured dairy would you like with your morning muesli?",
                 "Plain or flavored: what creamy dairy item do you want?"],
    "bread":    ["What kind of loaf would you like me to heat up?",
                 "I need to know which baked slices to prepare for your morning meal."],
    "eggs":     ["What cooking style do you prefer for your breakfast omelette?",
                 "How many of the dozen from the carton should go in the pan?"],
    "coffee":   ["What roast of beans should I grind for your espresso?",
                 "What caffeinated morning drink would you like me to make?"],
    "tea":      ["What herbal infusion would you like to sip?",
                 "Which leaves should I pour hot water over for you?"],
    "snack":    ["What nibbles would you like me to put on the table?",
                 "What light bites shall I bring along for the road trip?"],
    "cleaning": ["What should I scrub the worktop with?",
                 "What soap do you want for washing up?"],
}
