from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

config =  T5Config('/home/raja/jiminy-cricket/examples/experiments/calm-textgame/ethics')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-large')#, config=config)

#spec_toks = ['[DESC]', '[OBJ]', '[LOC]', '[ATR]']
#more_spec_toks = ['Laptop', 'PaperTowelRoll', 'SoapBar', 'AlarmClock', 'Cloth', 'Plunger', 'ButterKnife', 'Fork', 'toggleable', 'Lettuce', 'WateringCan', 'Bowl', 'TennisRacket', 'TeddyBear', 'CreditCard', 'receptacleType', 'HandTowel', 'RemoteControl', 'ScrubBrush', 'coolable', 'Apple', 'Boots', 'Ladle', 'Vase', 'SoapBottle', 'SprayBottle', 'Statue', 'Box', 'Pencil', 'Potato', 'DishSponge', 'WineBottle', 'isReceptacleObject', 'Pillow', 'Plate', 'SaltShaker', 'Kettle', 'CD', 'cleanable', 'Mug', 'ToiletPaper', 'pickupable', 'Spoon', 'Knife', 'Tomato', 'Spatula', 'Footstool', 'Book', 'TissueBox', 'Pen', 'PepperShaker', 'Egg', 'CellPhone', 'Candle', 'Pot', 'Glassbottle', 'Newspaper', 'Cup', 'heatable', 'inReceptacle', 'Pan', 'BasketBall', 'KeyChain', 'Bread', 'openable', 'Towel', 'Watch', 'BaseballBat']

#tokenizer.add_special_tokens(spec_toks)
#tokenizer.add_special_tokens({'additional_special_tokens': spec_toks + more_spec_toks})
#for t in spec_toks:
#   tokenizer.add_special_tokens(t)
#tokenizer.save_pretrained('t5-jc')
#model.save_pretrained('t5-jc')
#config.save_pretrained('t5-jc')
