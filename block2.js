// ==========================================
// GET FORM DATA
// ==========================================
const formData = $input.first().json;

const user = {
  name: formData['Full Name'] || 'Student',
  email: formData['Email'] || '',
  experience: (formData['Experience Level'] || 'Intermediate').split(' - ')[0].trim(),
  interests: formData['Interests'] || [],
  submittedAt: formData['submittedAt'] || new Date().toISOString()
};

console.log('User data:', JSON.stringify(user, null, 2));

// ==========================================
// ALL 30 COURSES - HARDCODED
// ==========================================
const courses = [
  // BEGINNER COURSES (5)
  {
    name: "Python Programming Fundamentals",
    topics: "Python syntax, Data structures, Functions, OOP, NumPy basics, Pandas introduction",
    duration: "4 weeks",
    difficulty: "Beginner",
    prerequisites: "None"
  },
  {
    name: "Mathematics for Machine Learning",
    topics: "Linear algebra, Calculus, Probability, Statistics, Optimization basics",
    duration: "6 weeks",
    difficulty: "Beginner",
    prerequisites: "Basic programming"
  },
  {
    name: "Introduction to Machine Learning",
    topics: "Supervised learning, Regression, Classification, Decision trees, Model evaluation, Cross-validation",
    duration: "8 weeks",
    difficulty: "Beginner",
    prerequisites: "Python programming"
  },
  {
    name: "Data Science Basics",
    topics: "Data cleaning, EDA, Visualization, Feature engineering, Scikit-learn basics",
    duration: "6 weeks",
    difficulty: "Beginner",
    prerequisites: "Python, Basic statistics"
  },
  {
    name: "AI Ethics and Responsible AI",
    topics: "AI bias, Fairness, Privacy, Transparency, Explainability, Regulatory compliance",
    duration: "4 weeks",
    difficulty: "Beginner",
    prerequisites: "None"
  },
  
  // INTERMEDIATE COURSES (10)
  {
    name: "Deep Learning Fundamentals",
    topics: "Neural network basics, Activation functions, Loss functions, Gradient descent, Backpropagation",
    duration: "6 weeks",
    difficulty: "Intermediate",
    prerequisites: "Python, Mathematics for ML, Introduction to ML"
  },
  {
    name: "Neural Networks and Backpropagation",
    topics: "Deep architectures, Weight initialization, Regularization, Dropout, Batch normalization",
    duration: "6 weeks",
    difficulty: "Intermediate",
    prerequisites: "Deep Learning Fundamentals"
  },
  {
    name: "Convolutional Neural Networks (CNNs)",
    topics: "Convolution operations, Pooling, CNN architectures (LeNet, AlexNet, VGG), Feature maps",
    duration: "8 weeks",
    difficulty: "Intermediate",
    prerequisites: "Neural Networks and Backpropagation"
  },
  {
    name: "Computer Vision Fundamentals",
    topics: "Image processing, Edge detection, Feature extraction, Classical CV vs Deep Learning",
    duration: "6 weeks",
    difficulty: "Intermediate",
    prerequisites: "Python, Basic image processing"
  },
  {
    name: "Image Classification with PyTorch",
    topics: "PyTorch fundamentals, Dataset creation, Training loops, ResNet, EfficientNet implementation",
    duration: "8 weeks",
    difficulty: "Intermediate",
    prerequisites: "Deep Learning Fundamentals, Python"
  },
  {
    name: "Object Detection and Recognition",
    topics: "YOLO, R-CNN family, SSD, RetinaNet, Anchor boxes, Non-max suppression, mAP metrics",
    duration: "8 weeks",
    difficulty: "Intermediate",
    prerequisites: "CNNs, Computer Vision Fundamentals"
  },
  {
    name: "Natural Language Processing Basics",
    topics: "Text preprocessing, Word embeddings, Sentiment analysis, Sequence models, Attention basics",
    duration: "6 weeks",
    difficulty: "Intermediate",
    prerequisites: "Python, Basic NLP concepts"
  },
  {
    name: "Recurrent Neural Networks (RNNs)",
    topics: "LSTM, GRU, Seq2seq models, Time series prediction, Language modeling",
    duration: "6 weeks",
    difficulty: "Intermediate",
    prerequisites: "Deep Learning Fundamentals"
  },
  {
    name: "Transfer Learning and Fine-tuning",
    topics: "Pre-trained models, ImageNet weights, Domain adaptation, Few-shot learning",
    duration: "4 weeks",
    difficulty: "Intermediate",
    prerequisites: "CNNs, PyTorch experience"
  },
  {
    name: "Data Augmentation Techniques",
    topics: "Rotation, Flipping, Color jittering, Mixup, CutMix, AutoAugment strategies",
    duration: "4 weeks",
    difficulty: "Intermediate",
    prerequisites: "Computer Vision Fundamentals"
  },
  
  // ADVANCED COURSES (15)
  {
    name: "Advanced Computer Vision",
    topics: "Semantic segmentation, Panoptic segmentation, Keypoint detection, Pose estimation, DeepLab, Mask R-CNN",
    duration: "10 weeks",
    difficulty: "Advanced",
    prerequisites: "CNNs, Object Detection"
  },
  {
    name: "Generative Adversarial Networks (GANs)",
    topics: "GAN architectures, StyleGAN, Pix2Pix, CycleGAN, Image synthesis, Super-resolution",
    duration: "8 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning Fundamentals, CNNs"
  },
  {
    name: "Image Segmentation and Instance Detection",
    topics: "U-Net, FCN, SegFormer, Instance segmentation, Medical image segmentation",
    duration: "8 weeks",
    difficulty: "Advanced",
    prerequisites: "CNNs, Object Detection"
  },
  {
    name: "Video Understanding and Action Recognition",
    topics: "Action recognition, Temporal modeling, 3D CNNs, Two-stream networks, SlowFast",
    duration: "8 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning, CNNs, RNNs"
  },
  {
    name: "3D Computer Vision and Point Clouds",
    topics: "3D object detection, Point cloud processing, PointNet, Voxel-based methods, LiDAR",
    duration: "10 weeks",
    difficulty: "Advanced",
    prerequisites: "Computer Vision Fundamentals, Linear algebra"
  },
  {
    name: "Vision Transformers (ViT)",
    topics: "Transformer architecture, Patch embeddings, Self-attention, DINO, Swin Transformer",
    duration: "8 weeks",
    difficulty: "Advanced",
    prerequisites: "CNNs, Attention mechanisms"
  },
  {
    name: "Self-Supervised Learning in Vision",
    topics: "SimCLR, MoCo, BYOL, SwAV, Contrastive learning, Self-distillation",
    duration: "6 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning, CNNs, PyTorch"
  },
  {
    name: "Multi-Modal Learning",
    topics: "Vision-Language models, CLIP, ALIGN, Image captioning, Visual Q&A",
    duration: "8 weeks",
    difficulty: "Advanced",
    prerequisites: "NLP Basics, Computer Vision"
  },
  {
    name: "Deep Learning for Medical Imaging",
    topics: "X-ray analysis, CT/MRI segmentation, Disease classification, FDA regulations",
    duration: "10 weeks",
    difficulty: "Advanced",
    prerequisites: "Advanced Computer Vision, Medical domain knowledge"
  },
  {
    name: "Reinforcement Learning Fundamentals",
    topics: "MDP, Q-learning, Policy gradients, DQN, Actor-Critic, PPO for robotics",
    duration: "10 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning, Probability theory"
  },
  {
    name: "MLOps and Model Deployment",
    topics: "Docker, Kubernetes, MLflow, Model serving, A/B testing, Monitoring, CI/CD pipelines",
    duration: "6 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning, Software engineering basics"
  },
  {
    name: "Edge AI and Mobile Vision",
    topics: "TensorFlow Lite, ONNX, Model compression, On-device inference, CoreML, Mobile deployment",
    duration: "6 weeks",
    difficulty: "Advanced",
    prerequisites: "Computer Vision, Mobile development"
  },
  {
    name: "Real-time Object Tracking",
    topics: "DeepSORT, Kalman filters, Multi-object tracking, Re-identification, ByteTrack",
    duration: "6 weeks",
    difficulty: "Advanced",
    prerequisites: "Object Detection, Kalman filters"
  },
  {
    name: "Neural Architecture Search",
    topics: "NAS algorithms, AutoML, DARTS, EfficientNet architecture search, Hardware-aware NAS",
    duration: "8 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning, AutoML concepts"
  },
  {
    name: "Model Optimization and Quantization",
    topics: "Pruning, Quantization, Knowledge distillation, TensorRT, INT8 inference",
    duration: "6 weeks",
    difficulty: "Advanced",
    prerequisites: "Deep Learning deployment experience"
  }
];

console.log('Total courses loaded:', courses.length);

// ==========================================
// RETURN COMBINED DATA
// ==========================================
return [{
  json: {
    user: user,
    availableCourses: courses
  }
}];
