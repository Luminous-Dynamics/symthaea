// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Music Education Hook
 *
 * Learning and education features:
 * - Interactive lessons
 * - AI practice assistant
 * - Ear training games
 * - Production tutorials
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Types
export interface Lesson {
  id: string;
  title: string;
  description: string;
  category: LessonCategory;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: number;  // minutes
  modules: LessonModule[];
  prerequisites?: string[];
  instructor?: Instructor;
  rating: number;
  enrolledCount: number;
  completionRate: number;
}

export type LessonCategory =
  | 'music-theory'
  | 'production'
  | 'mixing'
  | 'mastering'
  | 'sound-design'
  | 'djing'
  | 'songwriting'
  | 'arrangement'
  | 'instrument';

export interface LessonModule {
  id: string;
  title: string;
  type: 'video' | 'interactive' | 'quiz' | 'practice' | 'project';
  duration: number;
  content: ModuleContent;
  completed: boolean;
}

export interface ModuleContent {
  videoUrl?: string;
  interactiveComponent?: string;
  quiz?: QuizQuestion[];
  practiceTask?: PracticeTask;
  projectBrief?: ProjectBrief;
}

export interface QuizQuestion {
  id: string;
  type: 'multiple-choice' | 'audio-identify' | 'drag-drop' | 'fill-blank';
  question: string;
  audioUrl?: string;
  options?: string[];
  correctAnswer: string | string[];
  explanation: string;
}

export interface PracticeTask {
  id: string;
  title: string;
  instructions: string;
  audioReference?: string;
  targetBPM?: number;
  targetKey?: string;
  evaluationCriteria: EvaluationCriterion[];
}

export interface EvaluationCriterion {
  name: string;
  weight: number;
  description: string;
}

export interface ProjectBrief {
  title: string;
  description: string;
  requirements: string[];
  resourceFiles?: string[];
  submissionFormat: 'audio' | 'project-file' | 'video';
}

export interface Instructor {
  id: string;
  name: string;
  avatar: string;
  bio: string;
  credentials: string[];
  courseCount: number;
  studentCount: number;
  rating: number;
}

export interface EarTrainingExercise {
  id: string;
  type: EarTrainingType;
  difficulty: number;  // 1-10
  audioUrl: string;
  question: string;
  options: string[];
  correctAnswer: string;
  hintsAvailable: number;
}

export type EarTrainingType =
  | 'interval-recognition'
  | 'chord-recognition'
  | 'scale-recognition'
  | 'rhythm-dictation'
  | 'melody-dictation'
  | 'chord-progression'
  | 'relative-pitch'
  | 'absolute-pitch';

export interface PracticeSession {
  id: string;
  startedAt: Date;
  endedAt?: Date;
  type: 'ear-training' | 'rhythm' | 'theory' | 'production';
  exercises: ExerciseResult[];
  score: number;
  streak: number;
}

export interface ExerciseResult {
  exerciseId: string;
  correct: boolean;
  responseTime: number;
  attempts: number;
}

export interface UserProgress {
  lessonsCompleted: string[];
  lessonsInProgress: { lessonId: string; moduleIndex: number }[];
  totalPracticeTime: number;
  streakDays: number;
  longestStreak: number;
  skillLevels: Record<LessonCategory, number>;
  achievements: Achievement[];
  weeklyGoal: number;
  weeklyProgress: number;
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  unlockedAt: Date;
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
}

export interface AIFeedback {
  overall: string;
  strengths: string[];
  improvements: string[];
  tips: string[];
  score: number;
  detailedAnalysis?: {
    timing: number;
    pitch: number;
    dynamics: number;
    technique: number;
  };
}

export interface MusicEducationState {
  isLoading: boolean;
  lessons: Lesson[];
  currentLesson: Lesson | null;
  currentModuleIndex: number;
  progress: UserProgress | null;
  activePracticeSession: PracticeSession | null;
  currentExercise: EarTrainingExercise | null;
  aiFeedback: AIFeedback | null;
  error: string | null;
}

// Initial progress
const INITIAL_PROGRESS: UserProgress = {
  lessonsCompleted: [],
  lessonsInProgress: [],
  totalPracticeTime: 0,
  streakDays: 0,
  longestStreak: 0,
  skillLevels: {
    'music-theory': 0,
    'production': 0,
    'mixing': 0,
    'mastering': 0,
    'sound-design': 0,
    'djing': 0,
    'songwriting': 0,
    'arrangement': 0,
    'instrument': 0,
  },
  achievements: [],
  weeklyGoal: 60, // minutes
  weeklyProgress: 0,
};

export function useMusicEducation(userId?: string) {
  const [state, setState] = useState<MusicEducationState>({
    isLoading: false,
    lessons: [],
    currentLesson: null,
    currentModuleIndex: 0,
    progress: null,
    activePracticeSession: null,
    currentExercise: null,
    aiFeedback: null,
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const practiceStartTimeRef = useRef<number>(0);

  /**
   * Load user progress
   */
  const loadProgress = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Would load from API
      await new Promise(resolve => setTimeout(resolve, 300));

      setState(prev => ({
        ...prev,
        isLoading: false,
        progress: INITIAL_PROGRESS,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: 'Failed to load progress',
      }));
    }
  }, []);

  /**
   * Get lessons by category
   */
  const getLessons = useCallback(async (
    category?: LessonCategory,
    difficulty?: string
  ): Promise<Lesson[]> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 500));

      const lessons: Lesson[] = [
        {
          id: 'music-theory-101',
          title: 'Music Theory Fundamentals',
          description: 'Learn the building blocks of music: notes, scales, and chords',
          category: 'music-theory',
          difficulty: 'beginner',
          duration: 120,
          modules: [
            {
              id: 'mt-1-1',
              title: 'Introduction to Notes',
              type: 'video',
              duration: 15,
              content: { videoUrl: '/lessons/notes-intro.mp4' },
              completed: false,
            },
            {
              id: 'mt-1-2',
              title: 'The Major Scale',
              type: 'interactive',
              duration: 20,
              content: { interactiveComponent: 'ScaleExplorer' },
              completed: false,
            },
            {
              id: 'mt-1-3',
              title: 'Building Chords',
              type: 'video',
              duration: 25,
              content: { videoUrl: '/lessons/chords-intro.mp4' },
              completed: false,
            },
            {
              id: 'mt-1-4',
              title: 'Quiz: Fundamentals',
              type: 'quiz',
              duration: 10,
              content: {
                quiz: [
                  {
                    id: 'q1',
                    type: 'multiple-choice',
                    question: 'How many semitones are in an octave?',
                    options: ['10', '11', '12', '13'],
                    correctAnswer: '12',
                    explanation: 'An octave contains 12 semitones (half steps).',
                  },
                ],
              },
              completed: false,
            },
          ],
          instructor: {
            id: 'instructor-1',
            name: 'Sarah Chen',
            avatar: '/instructors/sarah.jpg',
            bio: 'Berklee graduate with 10+ years teaching experience',
            credentials: ['Berklee College of Music', 'Grammy-nominated producer'],
            courseCount: 12,
            studentCount: 50000,
            rating: 4.9,
          },
          rating: 4.8,
          enrolledCount: 25000,
          completionRate: 78,
        },
        {
          id: 'production-basics',
          title: 'Electronic Music Production 101',
          description: 'Create your first electronic track from scratch',
          category: 'production',
          difficulty: 'beginner',
          duration: 180,
          modules: [],
          rating: 4.7,
          enrolledCount: 18000,
          completionRate: 65,
        },
        {
          id: 'mixing-fundamentals',
          title: 'Mixing Fundamentals',
          description: 'Learn professional mixing techniques',
          category: 'mixing',
          difficulty: 'intermediate',
          duration: 240,
          modules: [],
          prerequisites: ['production-basics'],
          rating: 4.9,
          enrolledCount: 12000,
          completionRate: 72,
        },
      ];

      const filtered = lessons.filter(l => {
        if (category && l.category !== category) return false;
        if (difficulty && l.difficulty !== difficulty) return false;
        return true;
      });

      setState(prev => ({
        ...prev,
        isLoading: false,
        lessons: filtered,
      }));

      return filtered;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to load lessons' }));
      return [];
    }
  }, []);

  /**
   * Start a lesson
   */
  const startLesson = useCallback(async (lessonId: string): Promise<boolean> => {
    const lesson = state.lessons.find(l => l.id === lessonId);
    if (!lesson) {
      setState(prev => ({ ...prev, error: 'Lesson not found' }));
      return false;
    }

    setState(prev => ({
      ...prev,
      currentLesson: lesson,
      currentModuleIndex: 0,
    }));

    return true;
  }, [state.lessons]);

  /**
   * Complete current module and advance
   */
  const completeModule = useCallback(async (): Promise<boolean> => {
    if (!state.currentLesson) return false;

    const { currentLesson, currentModuleIndex } = state;
    const isLastModule = currentModuleIndex >= currentLesson.modules.length - 1;

    // Mark module as completed
    const updatedModules = [...currentLesson.modules];
    updatedModules[currentModuleIndex] = {
      ...updatedModules[currentModuleIndex],
      completed: true,
    };

    if (isLastModule) {
      // Lesson complete
      setState(prev => ({
        ...prev,
        currentLesson: null,
        currentModuleIndex: 0,
        progress: prev.progress
          ? {
              ...prev.progress,
              lessonsCompleted: [...prev.progress.lessonsCompleted, currentLesson.id],
              skillLevels: {
                ...prev.progress.skillLevels,
                [currentLesson.category]: prev.progress.skillLevels[currentLesson.category] + 10,
              },
            }
          : null,
      }));
    } else {
      // Advance to next module
      setState(prev => ({
        ...prev,
        currentLesson: { ...currentLesson, modules: updatedModules },
        currentModuleIndex: prev.currentModuleIndex + 1,
      }));
    }

    return true;
  }, [state.currentLesson, state.currentModuleIndex]);

  /**
   * Start ear training session
   */
  const startEarTraining = useCallback(async (
    type: EarTrainingType,
    difficulty: number = 5
  ): Promise<boolean> => {
    try {
      const session: PracticeSession = {
        id: `session-${Date.now()}`,
        startedAt: new Date(),
        type: 'ear-training',
        exercises: [],
        score: 0,
        streak: 0,
      };

      setState(prev => ({ ...prev, activePracticeSession: session }));
      practiceStartTimeRef.current = Date.now();

      // Generate first exercise
      await generateEarTrainingExercise(type, difficulty);

      return true;
    } catch (error) {
      setState(prev => ({ ...prev, error: 'Failed to start ear training' }));
      return false;
    }
  }, []);

  /**
   * Generate ear training exercise
   */
  const generateEarTrainingExercise = useCallback(async (
    type: EarTrainingType,
    difficulty: number
  ) => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 300));

      let exercise: EarTrainingExercise;

      switch (type) {
        case 'interval-recognition':
          exercise = generateIntervalExercise(difficulty);
          break;
        case 'chord-recognition':
          exercise = generateChordExercise(difficulty);
          break;
        case 'scale-recognition':
          exercise = generateScaleExercise(difficulty);
          break;
        default:
          exercise = generateIntervalExercise(difficulty);
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        currentExercise: exercise,
      }));
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to generate exercise' }));
    }
  }, []);

  /**
   * Submit answer for current exercise
   */
  const submitAnswer = useCallback(async (answer: string): Promise<{
    correct: boolean;
    explanation: string;
  }> => {
    if (!state.currentExercise || !state.activePracticeSession) {
      return { correct: false, explanation: '' };
    }

    const correct = answer === state.currentExercise.correctAnswer;
    const responseTime = Date.now() - practiceStartTimeRef.current;

    const result: ExerciseResult = {
      exerciseId: state.currentExercise.id,
      correct,
      responseTime,
      attempts: 1,
    };

    setState(prev => ({
      ...prev,
      activePracticeSession: prev.activePracticeSession
        ? {
            ...prev.activePracticeSession,
            exercises: [...prev.activePracticeSession.exercises, result],
            score: prev.activePracticeSession.score + (correct ? 10 : 0),
            streak: correct ? prev.activePracticeSession.streak + 1 : 0,
          }
        : null,
    }));

    return {
      correct,
      explanation: `The correct answer is ${state.currentExercise.correctAnswer}. ${
        correct ? 'Great job!' : 'Keep practicing!'
      }`,
    };
  }, [state.currentExercise, state.activePracticeSession]);

  /**
   * Get next exercise
   */
  const nextExercise = useCallback(async () => {
    if (!state.currentExercise) return;

    const type = state.currentExercise.type;
    const difficulty = state.currentExercise.difficulty;

    await generateEarTrainingExercise(type, difficulty);
    practiceStartTimeRef.current = Date.now();
  }, [state.currentExercise, generateEarTrainingExercise]);

  /**
   * End practice session
   */
  const endPracticeSession = useCallback(async (): Promise<PracticeSession | null> => {
    if (!state.activePracticeSession) return null;

    const session: PracticeSession = {
      ...state.activePracticeSession,
      endedAt: new Date(),
    };

    // Update progress
    const practiceMinutes = Math.round(
      (Date.now() - session.startedAt.getTime()) / 60000
    );

    setState(prev => ({
      ...prev,
      activePracticeSession: null,
      currentExercise: null,
      progress: prev.progress
        ? {
            ...prev.progress,
            totalPracticeTime: prev.progress.totalPracticeTime + practiceMinutes,
            weeklyProgress: prev.progress.weeklyProgress + practiceMinutes,
          }
        : null,
    }));

    return session;
  }, [state.activePracticeSession]);

  /**
   * Get AI feedback on recording
   */
  const getAIFeedback = useCallback(async (
    audioBuffer: AudioBuffer,
    task: PracticeTask
  ): Promise<AIFeedback> => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      await new Promise(resolve => setTimeout(resolve, 2000));

      const feedback: AIFeedback = {
        overall: 'Good effort! Your timing is solid, but there\'s room to improve dynamics.',
        strengths: [
          'Accurate rhythm throughout',
          'Good tempo consistency',
          'Clean note transitions',
        ],
        improvements: [
          'Add more dynamic variation',
          'Work on expression in louder passages',
          'Smooth out the crescendos',
        ],
        tips: [
          'Try practicing with a metronome at slower tempos first',
          'Listen to reference recordings for dynamic inspiration',
          'Record yourself to identify patterns',
        ],
        score: 78,
        detailedAnalysis: {
          timing: 85,
          pitch: 72,
          dynamics: 68,
          technique: 80,
        },
      };

      setState(prev => ({
        ...prev,
        isLoading: false,
        aiFeedback: feedback,
      }));

      return feedback;
    } catch (error) {
      setState(prev => ({ ...prev, isLoading: false, error: 'Failed to analyze' }));
      throw error;
    }
  }, []);

  /**
   * Get recommended lessons
   */
  const getRecommendedLessons = useCallback(async (): Promise<Lesson[]> => {
    if (!state.progress) return [];

    // Find weakest skill areas
    const skills = state.progress.skillLevels;
    const weakestCategory = Object.entries(skills).sort(([, a], [, b]) => a - b)[0][0];

    return getLessons(weakestCategory as LessonCategory);
  }, [state.progress, getLessons]);

  // Load progress on mount
  useEffect(() => {
    loadProgress();
  }, [loadProgress]);

  return {
    ...state,
    loadProgress,
    getLessons,
    startLesson,
    completeModule,
    startEarTraining,
    submitAnswer,
    nextExercise,
    endPracticeSession,
    getAIFeedback,
    getRecommendedLessons,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function generateIntervalExercise(difficulty: number): EarTrainingExercise {
  const intervals = ['m2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7', 'P8'];
  const availableIntervals = intervals.slice(0, Math.min(3 + difficulty, intervals.length));
  const correct = availableIntervals[Math.floor(Math.random() * availableIntervals.length)];

  return {
    id: `interval-${Date.now()}`,
    type: 'interval-recognition',
    difficulty,
    audioUrl: `/audio/intervals/${correct}.mp3`,
    question: 'What interval do you hear?',
    options: availableIntervals,
    correctAnswer: correct,
    hintsAvailable: 3 - Math.floor(difficulty / 4),
  };
}

function generateChordExercise(difficulty: number): EarTrainingExercise {
  const chords = ['major', 'minor', 'dim', 'aug', 'maj7', 'min7', 'dom7', 'dim7'];
  const availableChords = chords.slice(0, Math.min(2 + difficulty, chords.length));
  const correct = availableChords[Math.floor(Math.random() * availableChords.length)];

  return {
    id: `chord-${Date.now()}`,
    type: 'chord-recognition',
    difficulty,
    audioUrl: `/audio/chords/${correct}.mp3`,
    question: 'What chord quality do you hear?',
    options: availableChords,
    correctAnswer: correct,
    hintsAvailable: 3 - Math.floor(difficulty / 4),
  };
}

function generateScaleExercise(difficulty: number): EarTrainingExercise {
  const scales = ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian'];
  const availableScales = scales.slice(0, Math.min(2 + difficulty, scales.length));
  const correct = availableScales[Math.floor(Math.random() * availableScales.length)];

  return {
    id: `scale-${Date.now()}`,
    type: 'scale-recognition',
    difficulty,
    audioUrl: `/audio/scales/${correct}.mp3`,
    question: 'What scale do you hear?',
    options: availableScales,
    correctAnswer: correct,
    hintsAvailable: 3 - Math.floor(difficulty / 4),
  };
}

export default useMusicEducation;
