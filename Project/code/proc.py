def detect_ingrs(recipe, vocab):

    try:
        ingr_names = [ingr['text'] for ingr in recipe['ingredients']
                      if ingr['text']]
    except Exception:
        ingr_names = []
        print('Could not load ingredients! Moving on...')

    detected = set()
    for name in ingr_names:
        name = name.replace(' ', '_')
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)

    return list(detected) + [vocab['</i>']]
