// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CoursesPage Component
 *
 * Main page for course discovery with search, filters, and grid of course cards.
 * Now connected to Holochain via useCourses hook.
 */

import React, { useState, useMemo, useCallback } from 'react';
import { CourseCard } from '../components/CourseCard';
import { CourseFilters, FilterState } from '../components/CourseFilters';
import { LoadingCard } from '../components/LoadingSkeleton';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { useToast } from '../contexts/ToastContext';
import { useCourses, DisplayCourse } from '../hooks/useCourses';
import {
  SUCCESS_MESSAGES,
  INFO_MESSAGES,
} from '../config/constants';

// Re-export Course type for compatibility with CourseCard
export type Course = DisplayCourse;

export const CoursesPage: React.FC = () => {
  const { toast } = useToast();
  const { courses, loading, error, isReal, refresh } = useCourses();

  const [filters, setFilters] = useState<FilterState>({
    searchQuery: '',
    difficulty: 'all',
    tags: [],
    sortBy: 'popular',
  });

  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);

  // Extract all unique tags from courses
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    courses.forEach((course) => {
      course.tags.forEach((tag) => tagSet.add(tag));
    });
    return Array.from(tagSet).sort();
  }, [courses]);

  // Filter and sort courses
  const filteredCourses = useMemo(() => {
    let result = [...courses];

    // Apply search query
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      result = result.filter(
        (course) =>
          course.title.toLowerCase().includes(query) ||
          course.description.toLowerCase().includes(query) ||
          course.instructor.toLowerCase().includes(query) ||
          course.tags.some((tag) => tag.toLowerCase().includes(query))
      );
    }

    // Apply difficulty filter
    if (filters.difficulty !== 'all') {
      result = result.filter(
        (course) => course.difficulty.toLowerCase() === filters.difficulty.toLowerCase()
      );
    }

    // Apply tag filters
    if (filters.tags.length > 0) {
      result = result.filter((course) =>
        filters.tags.some((tag) => course.tags.includes(tag))
      );
    }

    // Apply sorting
    switch (filters.sortBy) {
      case 'popular':
        result.sort((a, b) => (b.enrollment_count || 0) - (a.enrollment_count || 0));
        break;
      case 'recent':
        result.sort(
          (a, b) =>
            new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()
        );
        break;
      case 'title':
        result.sort((a, b) => a.title.localeCompare(b.title));
        break;
    }

    return result;
  }, [courses, filters]);

  const handleCourseClick = useCallback((course: Course) => {
    setSelectedCourse(course);
    toast.info(INFO_MESSAGES.VIEWING_COURSE(course.title));
  }, [toast]);

  const handleCloseModal = useCallback(() => {
    setSelectedCourse(null);
  }, []);

  const handleClearFilters = useCallback(() => {
    setFilters({
      searchQuery: '',
      difficulty: 'all',
      tags: [],
      sortBy: 'popular',
    });
    toast.info(SUCCESS_MESSAGES.FILTERS_CLEARED);
  }, [toast]);

  // Show loading state
  if (loading) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#111827' }}>
            Discover Courses
          </h1>
          <p style={{ margin: '8px 0 0 0', fontSize: '16px', color: '#6b7280' }}>
            Loading courses...
          </p>
        </div>

        {/* Loading skeleton grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: '20px',
          }}
        >
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <LoadingCard key={i} />
          ))}
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#111827' }}>
            Discover Courses
          </h1>
        </div>

        <ErrorState
          title="Failed to Load Courses"
          message={error}
          onRetry={() => refresh()}
        />
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#111827' }}>
            Discover Courses
          </h1>
          {isReal && (
            <span
              style={{
                padding: '4px 8px',
                fontSize: '11px',
                fontWeight: '600',
                backgroundColor: '#ecfdf5',
                color: '#059669',
                borderRadius: '4px',
                textTransform: 'uppercase',
              }}
            >
              Live Data
            </span>
          )}
        </div>
        <p style={{ margin: '8px 0 0 0', fontSize: '16px', color: '#6b7280' }}>
          Browse {courses.length} courses from expert instructors
          {isReal && ' (connected to Holochain)'}
        </p>
      </div>

      {/* Filters */}
      <CourseFilters onFilterChange={setFilters} availableTags={allTags} />

      {/* Results Count */}
      <div style={{ marginBottom: '16px', fontSize: '14px', color: '#6b7280' }}>
        {filteredCourses.length === courses.length ? (
          <span>Showing all {filteredCourses.length} courses</span>
        ) : (
          <span>
            Found {filteredCourses.length} course{filteredCourses.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Course Grid */}
      {filteredCourses.length > 0 ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: '20px',
          }}
        >
          {filteredCourses.map((course) => (
            <CourseCard key={course.course_id} course={course} onClick={handleCourseClick} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon="🔍"
          title="No courses found"
          description="We couldn't find any courses matching your filters. Try adjusting your search criteria or browse all available courses."
          action={{
            label: "Clear Filters",
            onClick: handleClearFilters
          }}
          secondaryAction={{
            label: "View All Courses",
            onClick: () => {
              handleClearFilters();
              toast.info(INFO_MESSAGES.SHOWING_ALL_COURSES);
            }
          }}
        />
      )}

      {/* Course Detail Modal */}
      {selectedCourse && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '24px',
          }}
          onClick={handleCloseModal}
        >
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              maxWidth: '600px',
              width: '100%',
              maxHeight: '80vh',
              overflow: 'auto',
              padding: '32px',
              position: 'relative',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={handleCloseModal}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                cursor: 'pointer',
                fontSize: '18px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              ×
            </button>

            {/* Course Details */}
            <h2 style={{ margin: '0 0 8px 0', fontSize: '24px', fontWeight: '700' }}>
              {selectedCourse.title}
            </h2>
            <p style={{ margin: '0 0 24px 0', fontSize: '14px', color: '#6b7280' }}>
              by {selectedCourse.instructor}
            </p>

            <p style={{ margin: '0 0 24px 0', fontSize: '15px', lineHeight: '1.6' }}>
              {selectedCourse.description}
            </p>

            {/* Modules */}
            <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: '600' }}>
              Course Modules
            </h3>
            <div style={{ marginBottom: '24px' }}>
              {selectedCourse.syllabus.modules.map((module, index) => (
                <div
                  key={module.module_id}
                  style={{
                    padding: '12px',
                    marginBottom: '8px',
                    backgroundColor: '#f9fafb',
                    borderRadius: '6px',
                  }}
                >
                  <div style={{ fontWeight: '500', marginBottom: '4px' }}>
                    {index + 1}. {module.title}
                  </div>
                  <div style={{ fontSize: '13px', color: '#6b7280' }}>
                    ⏱️ {module.duration_hours} hours
                  </div>
                </div>
              ))}
            </div>

            {/* Enroll Button */}
            <button
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '16px',
                fontWeight: '600',
                color: '#ffffff',
                backgroundColor: '#3b82f6',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#2563eb';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#3b82f6';
              }}
              onClick={() => alert(INFO_MESSAGES.ENROLLMENT_COMING_SOON)}
            >
              Enroll Now
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
