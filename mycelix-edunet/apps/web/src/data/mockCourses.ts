// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mock course data
 */

export interface Course {
  course_id: string;
  title: string;
  description: string;
  instructor: string;
  syllabus: {
    modules: Module[];
    learning_outcomes?: string[];
    prerequisites?: string[];
  };
  tags: string[];
  difficulty: string;
  enrollment_count?: number;
  created_at?: string;
}

export interface Module {
  module_id: string;
  title: string;
  description?: string;
  duration_hours: number;
  learning_outcomes?: string[];
}

export const mockCourses: Course[] = [
  {
    course_id: 'rust-fundamentals-2025',
    title: 'Rust Fundamentals',
    description: 'Master the Rust programming language from basics to advanced concepts',
    instructor: 'Prof. Taylor Chen',
    syllabus: {
      modules: [
        {
          module_id: '1',
          title: 'Introduction to Rust',
          duration_hours: 3,
          learning_outcomes: ['Understand Rust syntax', 'Set up development environment'],
        },
        {
          module_id: '2',
          title: 'Ownership and Borrowing',
          duration_hours: 5,
          learning_outcomes: ['Master ownership rules', 'Understand borrowing and lifetimes'],
        },
        {
          module_id: '3',
          title: 'Structs and Enums',
          duration_hours: 4,
          learning_outcomes: ['Create custom data types', 'Use pattern matching'],
        },
        {
          module_id: '4',
          title: 'Error Handling',
          duration_hours: 3,
          learning_outcomes: ['Handle errors with Result', 'Use Option type effectively'],
        },
        {
          module_id: '5',
          title: 'Concurrency',
          duration_hours: 6,
          learning_outcomes: ['Write concurrent programs', 'Understand thread safety'],
        },
      ],
      prerequisites: ['Basic programming knowledge'],
    },
    tags: ['programming', 'rust', 'systems'],
    difficulty: 'intermediate',
    enrollment_count: 342,
    created_at: '2025-01-15T00:00:00Z',
  },
  {
    course_id: 'spanish-beginner-2025',
    title: 'Spanish for Beginners',
    description: 'Learn Spanish from scratch with interactive lessons',
    instructor: 'Prof. María González',
    syllabus: {
      modules: [
        {
          module_id: '1',
          title: 'Greetings and Introductions',
          duration_hours: 2,
        },
        {
          module_id: '2',
          title: 'Numbers and Colors',
          duration_hours: 2,
        },
        {
          module_id: '3',
          title: 'Family and Relationships',
          duration_hours: 3,
        },
        {
          module_id: '4',
          title: 'Present Tense Verbs',
          duration_hours: 4,
        },
      ],
    },
    tags: ['language', 'spanish', 'beginner'],
    difficulty: 'beginner',
    enrollment_count: 587,
    created_at: '2025-02-01T00:00:00Z',
  },
  {
    course_id: 'machine-learning-intro',
    title: 'Machine Learning Fundamentals',
    description: 'Introduction to ML concepts and practical applications',
    instructor: 'Dr. Sarah Kumar',
    syllabus: {
      modules: [
        {
          module_id: '1',
          title: 'What is Machine Learning?',
          duration_hours: 2,
        },
        {
          module_id: '2',
          title: 'Supervised Learning',
          duration_hours: 6,
        },
        {
          module_id: '3',
          title: 'Unsupervised Learning',
          duration_hours: 5,
        },
        {
          module_id: '4',
          title: 'Neural Networks Basics',
          duration_hours: 8,
        },
      ],
      prerequisites: ['Python programming', 'Basic statistics'],
    },
    tags: ['ai', 'ml', 'python', 'advanced'],
    difficulty: 'advanced',
    enrollment_count: 234,
    created_at: '2025-03-10T00:00:00Z',
  },
  {
    course_id: 'web3-dev',
    title: 'Web3 Development Basics',
    description: 'Build decentralized applications on blockchain',
    instructor: 'Alex Rivera',
    syllabus: {
      modules: [
        {
          module_id: '1',
          title: 'Blockchain Fundamentals',
          duration_hours: 3,
        },
        {
          module_id: '2',
          title: 'Smart Contracts',
          duration_hours: 6,
        },
        {
          module_id: '3',
          title: 'DApp Development',
          duration_hours: 8,
        },
      ],
      prerequisites: ['JavaScript', 'Basic cryptography'],
    },
    tags: ['web3', 'blockchain', 'solidity'],
    difficulty: 'intermediate',
    enrollment_count: 198,
    created_at: '2025-04-05T00:00:00Z',
  },
  {
    course_id: 'data-structures',
    title: 'Advanced Data Structures',
    description: 'Master complex data structures and algorithms',
    instructor: 'Prof. James Lee',
    syllabus: {
      modules: [
        {
          module_id: '1',
          title: 'Trees and Graphs',
          duration_hours: 5,
        },
        {
          module_id: '2',
          title: 'Hash Tables',
          duration_hours: 4,
        },
        {
          module_id: '3',
          title: 'Dynamic Programming',
          duration_hours: 6,
        },
        {
          module_id: '4',
          title: 'Advanced Algorithms',
          duration_hours: 7,
        },
      ],
      prerequisites: ['Data Structures Basics', 'Algorithm Analysis'],
    },
    tags: ['algorithms', 'programming', 'computer-science'],
    difficulty: 'advanced',
    enrollment_count: 156,
    created_at: '2025-05-20T00:00:00Z',
  },
  {
    course_id: 'cryptography',
    title: 'Applied Cryptography',
    description: 'Learn cryptographic principles and implementations',
    instructor: 'Dr. Emily Zhang',
    syllabus: {
      modules: [
        {
          module_id: '1',
          title: 'Symmetric Encryption',
          duration_hours: 4,
        },
        {
          module_id: '2',
          title: 'Public Key Cryptography',
          duration_hours: 5,
        },
        {
          module_id: '3',
          title: 'Hash Functions',
          duration_hours: 3,
        },
        {
          module_id: '4',
          title: 'Digital Signatures',
          duration_hours: 4,
        },
        {
          module_id: '5',
          title: 'Zero-Knowledge Proofs',
          duration_hours: 6,
        },
      ],
      prerequisites: ['Mathematics', 'Computer Science Fundamentals'],
    },
    tags: ['cryptography', 'security', 'mathematics'],
    difficulty: 'advanced',
    enrollment_count: 89,
    created_at: '2025-06-15T00:00:00Z',
  },
  {
    course_id: 'python-data-science',
    title: 'Python for Data Science',
    description: 'Master Python libraries for data analysis and visualization',
    instructor: 'Dr. Rachel Park',
    syllabus: {
      modules: [
        { module_id: '1', title: 'NumPy Essentials', duration_hours: 4 },
        { module_id: '2', title: 'Pandas for Data Manipulation', duration_hours: 6 },
        { module_id: '3', title: 'Matplotlib and Seaborn', duration_hours: 4 },
        { module_id: '4', title: 'Statistical Analysis', duration_hours: 5 },
      ],
      prerequisites: ['Python Basics'],
    },
    tags: ['python', 'data-science', 'analytics'],
    difficulty: 'intermediate',
    enrollment_count: 412,
    created_at: '2025-01-20T00:00:00Z',
  },
  {
    course_id: 'react-modern-web',
    title: 'Modern React Development',
    description: 'Build production-ready React applications with hooks and TypeScript',
    instructor: 'Jordan Mitchell',
    syllabus: {
      modules: [
        { module_id: '1', title: 'React Hooks Deep Dive', duration_hours: 5 },
        { module_id: '2', title: 'TypeScript with React', duration_hours: 4 },
        { module_id: '3', title: 'State Management (Redux, Zustand)', duration_hours: 6 },
        { module_id: '4', title: 'Testing React Apps', duration_hours: 4 },
        { module_id: '5', title: 'Performance Optimization', duration_hours: 3 },
      ],
      prerequisites: ['JavaScript ES6+', 'HTML/CSS'],
    },
    tags: ['react', 'javascript', 'web-development', 'typescript'],
    difficulty: 'intermediate',
    enrollment_count: 523,
    created_at: '2025-02-10T00:00:00Z',
  },
  {
    course_id: 'holochain-dev',
    title: 'Holochain Application Development',
    description: 'Build agent-centric distributed applications on Holochain',
    instructor: 'Dr. Marcus Johnson',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Holochain Architecture', duration_hours: 3 },
        { module_id: '2', title: 'Zome Development in Rust', duration_hours: 8 },
        { module_id: '3', title: 'DHT and Validation Rules', duration_hours: 6 },
        { module_id: '4', title: 'Building hApp UIs', duration_hours: 5 },
      ],
      prerequisites: ['Rust Fundamentals', 'Distributed Systems'],
    },
    tags: ['holochain', 'distributed-systems', 'rust', 'web3'],
    difficulty: 'advanced',
    enrollment_count: 147,
    created_at: '2025-03-01T00:00:00Z',
  },
  {
    course_id: 'linear-algebra',
    title: 'Linear Algebra for ML',
    description: 'Mathematical foundations for machine learning and AI',
    instructor: 'Prof. Lisa Wang',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Vectors and Matrices', duration_hours: 5 },
        { module_id: '2', title: 'Eigenvalues and Eigenvectors', duration_hours: 6 },
        { module_id: '3', title: 'Matrix Decomposition', duration_hours: 5 },
        { module_id: '4', title: 'Applications in ML', duration_hours: 4 },
      ],
      prerequisites: ['Calculus I'],
    },
    tags: ['mathematics', 'linear-algebra', 'ml'],
    difficulty: 'intermediate',
    enrollment_count: 289,
    created_at: '2025-01-25T00:00:00Z',
  },
  {
    course_id: 'node-backend',
    title: 'Node.js Backend Development',
    description: 'Build scalable server-side applications with Node.js and Express',
    instructor: 'Chris Anderson',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Node.js Fundamentals', duration_hours: 4 },
        { module_id: '2', title: 'Express.js Framework', duration_hours: 5 },
        { module_id: '3', title: 'Database Integration (MongoDB, PostgreSQL)', duration_hours: 6 },
        { module_id: '4', title: 'Authentication and Security', duration_hours: 5 },
        { module_id: '5', title: 'Deployment and Scaling', duration_hours: 4 },
      ],
      prerequisites: ['JavaScript Intermediate'],
    },
    tags: ['node', 'backend', 'javascript', 'express'],
    difficulty: 'intermediate',
    enrollment_count: 445,
    created_at: '2025-02-15T00:00:00Z',
  },
  {
    course_id: 'ethical-hacking',
    title: 'Ethical Hacking & Penetration Testing',
    description: 'Learn security testing methodologies and tools',
    instructor: 'Alex Cipher',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Reconnaissance and Scanning', duration_hours: 4 },
        { module_id: '2', title: 'Vulnerability Assessment', duration_hours: 5 },
        { module_id: '3', title: 'Web Application Security', duration_hours: 6 },
        { module_id: '4', title: 'Network Penetration', duration_hours: 5 },
        { module_id: '5', title: 'Reporting and Remediation', duration_hours: 3 },
      ],
      prerequisites: ['Networking Basics', 'Linux Command Line'],
    },
    tags: ['security', 'hacking', 'penetration-testing'],
    difficulty: 'advanced',
    enrollment_count: 201,
    created_at: '2025-04-01T00:00:00Z',
  },
  {
    course_id: 'deep-learning',
    title: 'Deep Learning with PyTorch',
    description: 'Advanced neural networks and deep learning techniques',
    instructor: 'Dr. Nina Patel',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Neural Network Foundations', duration_hours: 5 },
        { module_id: '2', title: 'Convolutional Neural Networks (CNNs)', duration_hours: 7 },
        { module_id: '3', title: 'Recurrent Neural Networks (RNNs)', duration_hours: 6 },
        { module_id: '4', title: 'Transformers and Attention', duration_hours: 8 },
        { module_id: '5', title: 'GANs and Advanced Architectures', duration_hours: 6 },
      ],
      prerequisites: ['Machine Learning Fundamentals', 'Python', 'Linear Algebra'],
    },
    tags: ['deep-learning', 'ai', 'pytorch', 'neural-networks'],
    difficulty: 'advanced',
    enrollment_count: 178,
    created_at: '2025-03-20T00:00:00Z',
  },
  {
    course_id: 'golang-concurrency',
    title: 'Concurrent Programming in Go',
    description: 'Master goroutines, channels, and concurrent patterns in Go',
    instructor: 'Sam Williams',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Go Language Basics', duration_hours: 3 },
        { module_id: '2', title: 'Goroutines and Channels', duration_hours: 5 },
        { module_id: '3', title: 'Concurrency Patterns', duration_hours: 6 },
        { module_id: '4', title: 'Sync Package and Mutexes', duration_hours: 4 },
        { module_id: '5', title: 'Building Concurrent Applications', duration_hours: 6 },
      ],
      prerequisites: ['Programming Fundamentals'],
    },
    tags: ['golang', 'concurrency', 'systems-programming'],
    difficulty: 'intermediate',
    enrollment_count: 267,
    created_at: '2025-02-28T00:00:00Z',
  },
  {
    course_id: 'css-advanced',
    title: 'Advanced CSS & Animations',
    description: 'Master modern CSS techniques, Grid, Flexbox, and animations',
    instructor: 'Maya Rodriguez',
    syllabus: {
      modules: [
        { module_id: '1', title: 'CSS Grid Mastery', duration_hours: 4 },
        { module_id: '2', title: 'Flexbox Deep Dive', duration_hours: 3 },
        { module_id: '3', title: 'CSS Animations and Transitions', duration_hours: 5 },
        { module_id: '4', title: 'Responsive Design Patterns', duration_hours: 4 },
        { module_id: '5', title: 'CSS Variables and Custom Properties', duration_hours: 3 },
      ],
      prerequisites: ['HTML/CSS Basics'],
    },
    tags: ['css', 'web-design', 'animations', 'responsive'],
    difficulty: 'intermediate',
    enrollment_count: 394,
    created_at: '2025-01-18T00:00:00Z',
  },
  {
    course_id: 'federated-learning',
    title: 'Federated Learning Systems',
    description: 'Privacy-preserving distributed machine learning',
    instructor: 'Prof. David Kim',
    syllabus: {
      modules: [
        { module_id: '1', title: 'FL Fundamentals', duration_hours: 4 },
        { module_id: '2', title: 'Differential Privacy', duration_hours: 6 },
        { module_id: '3', title: 'Aggregation Algorithms', duration_hours: 5 },
        { module_id: '4', title: 'Byzantine-Robust FL', duration_hours: 6 },
        { module_id: '5', title: 'Real-World Applications', duration_hours: 4 },
      ],
      prerequisites: ['Machine Learning', 'Distributed Systems'],
    },
    tags: ['federated-learning', 'privacy', 'ml', 'distributed'],
    difficulty: 'advanced',
    enrollment_count: 123,
    created_at: '2025-04-10T00:00:00Z',
  },
  {
    course_id: 'vue-composition',
    title: 'Vue 3 with Composition API',
    description: 'Modern Vue.js development with Composition API and TypeScript',
    instructor: 'Emma Chen',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Vue 3 Fundamentals', duration_hours: 4 },
        { module_id: '2', title: 'Composition API Deep Dive', duration_hours: 6 },
        { module_id: '3', title: 'State Management with Pinia', duration_hours: 5 },
        { module_id: '4', title: 'Vue Router and Routing', duration_hours: 4 },
      ],
      prerequisites: ['JavaScript ES6+'],
    },
    tags: ['vue', 'javascript', 'web-development', 'composition-api'],
    difficulty: 'intermediate',
    enrollment_count: 298,
    created_at: '2025-02-05T00:00:00Z',
  },
  {
    course_id: 'discrete-math',
    title: 'Discrete Mathematics for CS',
    description: 'Logic, set theory, graphs, and combinatorics for computer science',
    instructor: 'Prof. Alan Turing Jr.',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Logic and Proofs', duration_hours: 5 },
        { module_id: '2', title: 'Set Theory', duration_hours: 4 },
        { module_id: '3', title: 'Graph Theory', duration_hours: 6 },
        { module_id: '4', title: 'Combinatorics', duration_hours: 5 },
        { module_id: '5', title: 'Number Theory', duration_hours: 4 },
      ],
      prerequisites: ['High School Mathematics'],
    },
    tags: ['mathematics', 'discrete-math', 'computer-science'],
    difficulty: 'intermediate',
    enrollment_count: 312,
    created_at: '2025-01-30T00:00:00Z',
  },
  {
    course_id: 'docker-kubernetes',
    title: 'Docker and Kubernetes Essentials',
    description: 'Container orchestration and cloud-native deployment',
    instructor: 'Michael Santos',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Docker Fundamentals', duration_hours: 5 },
        { module_id: '2', title: 'Multi-Container Applications', duration_hours: 4 },
        { module_id: '3', title: 'Kubernetes Architecture', duration_hours: 6 },
        { module_id: '4', title: 'Deployments and Services', duration_hours: 5 },
        { module_id: '5', title: 'Production Best Practices', duration_hours: 4 },
      ],
      prerequisites: ['Linux Basics', 'Networking'],
    },
    tags: ['docker', 'kubernetes', 'devops', 'cloud'],
    difficulty: 'intermediate',
    enrollment_count: 376,
    created_at: '2025-03-15T00:00:00Z',
  },
  {
    course_id: 'solidity-smart-contracts',
    title: 'Solidity Smart Contract Development',
    description: 'Build and deploy secure Ethereum smart contracts',
    instructor: 'Dr. Satoshi Nakamoto III',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Solidity Basics', duration_hours: 4 },
        { module_id: '2', title: 'Smart Contract Patterns', duration_hours: 6 },
        { module_id: '3', title: 'Security Best Practices', duration_hours: 5 },
        { module_id: '4', title: 'Testing and Deployment', duration_hours: 5 },
        { module_id: '5', title: 'DeFi Protocols', duration_hours: 6 },
      ],
      prerequisites: ['JavaScript', 'Blockchain Fundamentals'],
    },
    tags: ['solidity', 'ethereum', 'smart-contracts', 'web3'],
    difficulty: 'advanced',
    enrollment_count: 189,
    created_at: '2025-04-20T00:00:00Z',
  },
  {
    course_id: 'typescript-advanced',
    title: 'Advanced TypeScript Patterns',
    description: 'Master type system, generics, and advanced TypeScript features',
    instructor: 'Olivia Thompson',
    syllabus: {
      modules: [
        { module_id: '1', title: 'Type System Deep Dive', duration_hours: 5 },
        { module_id: '2', title: 'Generics and Utility Types', duration_hours: 6 },
        { module_id: '3', title: 'Conditional Types', duration_hours: 4 },
        { module_id: '4', title: 'Decorators and Metadata', duration_hours: 5 },
      ],
      prerequisites: ['TypeScript Basics', 'JavaScript ES6+'],
    },
    tags: ['typescript', 'javascript', 'programming'],
    difficulty: 'advanced',
    enrollment_count: 245,
    created_at: '2025-02-22T00:00:00Z',
  },
];

export function getCourseById(id: string): Course | undefined {
  return mockCourses.find(c => c.course_id === id);
}

export function getCoursesByTag(tag: string): Course[] {
  return mockCourses.filter(c => c.tags.includes(tag.toLowerCase()));
}

export function getCoursesByDifficulty(difficulty: string): Course[] {
  return mockCourses.filter(c => c.difficulty === difficulty.toLowerCase());
}

export function searchCourses(query: string): Course[] {
  const lowerQuery = query.toLowerCase();
  return mockCourses.filter(
    c =>
      c.title.toLowerCase().includes(lowerQuery) ||
      c.description.toLowerCase().includes(lowerQuery) ||
      c.tags.some(tag => tag.includes(lowerQuery))
  );
}
