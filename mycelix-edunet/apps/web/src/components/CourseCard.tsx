// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CourseCard Component
 *
 * Displays a course in a card format with title, description, metadata, and enrollment info
 * Memoized to prevent unnecessary re-renders when props haven't changed
 */

import React from 'react';
import type { Course } from '../data/mockCourses';

interface CourseCardProps {
  course: Course;
  onClick?: (course: Course) => void;
}

export const CourseCard = React.memo<CourseCardProps>(({ course, onClick }) => {
  const totalHours = course.syllabus.modules.reduce(
    (sum, m) => sum + m.duration_hours,
    0
  );

  const handleClick = () => {
    if (onClick) {
      onClick(course);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (onClick && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onClick(course);
    }
  };

  const getDifficultyColor = (difficulty: string): string => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return '#10b981'; // green
      case 'intermediate':
        return '#3b82f6'; // blue
      case 'advanced':
        return '#ef4444'; // red
      default:
        return '#6b7280'; // gray
    }
  };

  return (
    <div
      className="course-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={`${course.title} by ${course.instructor}. ${course.difficulty} level. ${totalHours} hours. ${course.enrollment_count} enrolled. Click for details.`}
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '20px',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        backgroundColor: '#ffffff',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      }}
      onMouseEnter={(e) => {
        if (onClick) {
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <h3
            style={{
              margin: 0,
              fontSize: '18px',
              fontWeight: '600',
              color: '#111827',
              marginBottom: '4px',
            }}
          >
            {course.title}
          </h3>
          <span
            style={{
              fontSize: '11px',
              fontWeight: '500',
              padding: '4px 8px',
              borderRadius: '4px',
              backgroundColor: getDifficultyColor(course.difficulty) + '20',
              color: getDifficultyColor(course.difficulty),
              textTransform: 'capitalize',
            }}
          >
            {course.difficulty}
          </span>
        </div>
        <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#6b7280' }}>
          {course.instructor}
        </p>
      </div>

      {/* Description */}
      <p
        style={{
          margin: '0 0 16px 0',
          fontSize: '14px',
          color: '#4b5563',
          lineHeight: '1.5',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {course.description}
      </p>

      {/* Tags */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '16px' }}>
        {course.tags.slice(0, 4).map((tag) => (
          <span
            key={tag}
            style={{
              fontSize: '12px',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: '#f3f4f6',
              color: '#4b5563',
            }}
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Metadata */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingTop: '16px',
          borderTop: '1px solid #e5e7eb',
        }}
      >
        <div style={{ display: 'flex', gap: '16px', fontSize: '13px', color: '#6b7280' }}>
          <span>📚 {course.syllabus.modules.length} modules</span>
          <span>⏱️ {totalHours}h</span>
        </div>
        {course.enrollment_count !== undefined && (
          <span style={{ fontSize: '13px', color: '#6b7280' }}>
            👥 {course.enrollment_count} enrolled
          </span>
        )}
      </div>
    </div>
  );
});
