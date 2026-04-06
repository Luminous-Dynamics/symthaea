// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useCourses Hook
 *
 * Custom hook for fetching and managing courses from Holochain.
 * Bridges between the Holochain Course type and the UI-friendly DisplayCourse type.
 */

import { useState, useEffect, useCallback } from 'react';
import { useHolochain } from '../contexts/HolochainContext';
import { Course as HolochainCourse } from '../types/zomes';
import { mockCourses, Course as MockCourse } from '../data/mockCourses';

// =============================================================================
// Types
// =============================================================================

/**
 * Display-friendly course type that combines Holochain data with UI extras
 */
export interface DisplayCourse {
  course_id: string;
  title: string;
  description: string;
  instructor: string;
  syllabus: {
    modules: Array<{
      module_id: string;
      title: string;
      description?: string;
      duration_hours: number;
      learning_outcomes?: string[];
    }>;
    learning_outcomes?: string[];
    prerequisites?: string[];
  };
  tags: string[];
  difficulty: string;
  enrollment_count?: number;
  created_at?: string;
  // Holochain-specific fields (optional)
  model_id?: string | null;
  creator?: string;
  isFromHolochain?: boolean;
}

export interface UseCoursesResult {
  /** List of courses */
  courses: DisplayCourse[];
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh courses from source */
  refresh: () => Promise<void>;
  /** Create a new course */
  createCourse: (course: Partial<DisplayCourse>) => Promise<string | null>;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Convert Holochain Course to DisplayCourse
 */
function holochainToDisplayCourse(course: HolochainCourse): DisplayCourse {
  return {
    course_id: course.course_id,
    title: course.title,
    description: course.description,
    instructor: course.creator, // Use creator as instructor
    syllabus: {
      modules: [], // Holochain Course doesn't have modules by default
      learning_outcomes: [],
      prerequisites: [],
    },
    tags: course.tags,
    difficulty: 'intermediate', // Default difficulty
    enrollment_count: 0,
    created_at: new Date(course.created_at * 1000).toISOString(),
    model_id: course.model_id,
    creator: course.creator,
    isFromHolochain: true,
  };
}

/**
 * Convert MockCourse to DisplayCourse
 */
function mockToDisplayCourse(course: MockCourse): DisplayCourse {
  return {
    ...course,
    isFromHolochain: false,
  };
}

// =============================================================================
// Hook
// =============================================================================

export function useCourses(): UseCoursesResult {
  const { client, status, isReal } = useHolochain();
  const [courses, setCourses] = useState<DisplayCourse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch courses from appropriate source
  const fetchCourses = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        // Fetch from real Holochain
        console.log('[useCourses] Fetching from Holochain...');
        const holochainCourses = await client.learning.list_courses();
        const displayCourses = holochainCourses.map(holochainToDisplayCourse);

        // If no courses from Holochain, merge with mock data for demo
        if (displayCourses.length === 0) {
          console.log('[useCourses] No Holochain courses, using mock data');
          setCourses(mockCourses.map(mockToDisplayCourse));
        } else {
          setCourses(displayCourses);
        }
      } else {
        // Use mock data
        console.log('[useCourses] Using mock data');
        // Simulate network delay for realistic UX
        await new Promise(resolve => setTimeout(resolve, 500));
        setCourses(mockCourses.map(mockToDisplayCourse));
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch courses';
      console.error('[useCourses] Error:', errorMessage);
      setError(errorMessage);

      // Fall back to mock data on error
      setCourses(mockCourses.map(mockToDisplayCourse));
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  // Fetch on mount and when connection status changes
  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchCourses();
    }
  }, [status, fetchCourses]);

  // Create a new course
  const createCourse = useCallback(async (courseData: Partial<DisplayCourse>): Promise<string | null> => {
    if (!client || status !== 'connected') {
      console.warn('[useCourses] Cannot create course - not connected');
      return null;
    }

    try {
      if (isReal) {
        // Create in Holochain
        const newCourse: HolochainCourse = {
          course_id: courseData.course_id || `course-${Date.now()}`,
          title: courseData.title || 'Untitled Course',
          description: courseData.description || '',
          creator: '', // Will be set by zome
          tags: courseData.tags || [],
          model_id: courseData.model_id || null,
          created_at: Math.floor(Date.now() / 1000),
          updated_at: Math.floor(Date.now() / 1000),
          metadata: null,
        };

        const actionHash = await client.learning.create_course(newCourse);
        console.log('[useCourses] Course created:', actionHash);

        // Refresh the list
        await fetchCourses();

        return newCourse.course_id;
      } else {
        // Mock creation
        const newCourse: DisplayCourse = {
          course_id: courseData.course_id || `mock-${Date.now()}`,
          title: courseData.title || 'Untitled Course',
          description: courseData.description || '',
          instructor: courseData.instructor || 'Anonymous',
          syllabus: courseData.syllabus || { modules: [] },
          tags: courseData.tags || [],
          difficulty: courseData.difficulty || 'beginner',
          enrollment_count: 0,
          created_at: new Date().toISOString(),
          isFromHolochain: false,
        };

        setCourses(prev => [newCourse, ...prev]);
        return newCourse.course_id;
      }
    } catch (err) {
      console.error('[useCourses] Failed to create course:', err);
      return null;
    }
  }, [client, status, isReal, fetchCourses]);

  return {
    courses,
    loading,
    error,
    isReal,
    refresh: fetchCourses,
    createCourse,
  };
}

// =============================================================================
// Export
// =============================================================================

export default useCourses;
