# SkAI - Project Proposal

### Project Description

After skateboarding made its debut at Tokyo 2021 Olympic Games, the skateboarding industry is becoming a promising industry and it will attract more and more young people all over the world to play this sport. But at the current time, skateboarding is still an unpopular sport, in some cities, skaters can't find a near skateboarding park to practice tricks let alone evaluate tricks they did. For this, we're going to use AI to help skaters to evaluate how well the trick they did. To be specific, skaters can upload their skateboarding videos to the system, then the system will recognize tricks in their video and generate a score by artificial intelligence.

### Data sources

We are planning to collect free skateboarding tricks (ollie or kickflip) images without copyright from websites like Unsplash or Istockphoto. The number of photos might be 200. Then we can use a tool like LabelImg to help us label our image data.

### Models used

Actually, videos consist of a chain of still images. Based on this, we can extract a bunch of unbroken sequences of images with a fixed time interval. Then use the R-CNN model to train our network through our data source. 

### Challenges

- The tradeoff between accuracy and performance.
- There are several different R-CNN subsets, like "Fast R-CNN", "Faster R-CNN", and "Mask R-CNN". We need to pick one of them. 
- The configuration of the network.
- How to evaluate a skateboarding trick.

### Metrics for Success

- A well-trained deep neural network with a not-bad performance.
- Uploaded videos can be cut into a chain of still photos, then classify this trick into a category with good probability.
- No unexpected errors.

### Team Members

- Ninglin Liu, ninglinl
- Jiaxiang Wu,
- Chuchu Wu,
- Panke Jing,
