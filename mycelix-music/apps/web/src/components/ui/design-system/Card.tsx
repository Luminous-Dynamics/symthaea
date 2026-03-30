// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { forwardRef } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const cardVariants = cva(
  'rounded-xl transition-all duration-200',
  {
    variants: {
      variant: {
        default:
          'bg-white border border-gray-200 shadow-sm dark:bg-gray-900 dark:border-gray-800',
        elevated:
          'bg-white shadow-lg dark:bg-gray-900',
        outlined:
          'bg-transparent border-2 border-gray-200 dark:border-gray-700',
        ghost:
          'bg-gray-50 dark:bg-gray-800/50',
        glass:
          'backdrop-blur-xl bg-white/10 border border-white/20 shadow-xl',
        gradient:
          'bg-gradient-to-br from-primary-500/10 to-secondary-500/10 border border-primary-200/50 dark:border-primary-800/50',
      },
      padding: {
        none: 'p-0',
        sm: 'p-3',
        md: 'p-4',
        lg: 'p-6',
        xl: 'p-8',
      },
      interactive: {
        true: 'cursor-pointer hover:shadow-md hover:border-primary-300 dark:hover:border-primary-700 active:scale-[0.99]',
      },
    },
    defaultVariants: {
      variant: 'default',
      padding: 'md',
    },
  }
);

export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {
  asChild?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, padding, interactive, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(cardVariants({ variant, padding, interactive }), className)}
        {...props}
      />
    );
  }
);

Card.displayName = 'Card';

// Card subcomponents
const CardHeader = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex flex-col space-y-1.5', className)}
    {...props}
  />
));
CardHeader.displayName = 'CardHeader';

const CardTitle = forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn('text-lg font-semibold leading-none tracking-tight', className)}
    {...props}
  />
));
CardTitle.displayName = 'CardTitle';

const CardDescription = forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn('text-sm text-gray-500 dark:text-gray-400', className)}
    {...props}
  />
));
CardDescription.displayName = 'CardDescription';

const CardContent = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('', className)} {...props} />
));
CardContent.displayName = 'CardContent';

const CardFooter = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex items-center pt-4', className)}
    {...props}
  />
));
CardFooter.displayName = 'CardFooter';

// Media Card for albums/tracks
export interface MediaCardProps extends CardProps {
  image?: string;
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  aspectRatio?: 'square' | 'video' | 'portrait';
}

const MediaCard = forwardRef<HTMLDivElement, MediaCardProps>(
  (
    {
      className,
      image,
      title,
      subtitle,
      badge,
      actions,
      aspectRatio = 'square',
      interactive = true,
      ...props
    },
    ref
  ) => {
    const aspectClasses = {
      square: 'aspect-square',
      video: 'aspect-video',
      portrait: 'aspect-[3/4]',
    };

    return (
      <Card
        ref={ref}
        className={cn('overflow-hidden group', className)}
        padding="none"
        interactive={interactive}
        {...props}
      >
        <div className={cn('relative overflow-hidden', aspectClasses[aspectRatio])}>
          {image ? (
            <img
              src={image}
              alt={title}
              className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
            />
          ) : (
            <div className="h-full w-full bg-gradient-to-br from-primary-500/20 to-secondary-500/20 flex items-center justify-center">
              <span className="text-4xl text-gray-400">🎵</span>
            </div>
          )}
          {badge && (
            <div className="absolute top-2 left-2">{badge}</div>
          )}
          {actions && (
            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center gap-2">
              {actions}
            </div>
          )}
        </div>
        <div className="p-3">
          <h4 className="font-medium text-gray-900 dark:text-white truncate">
            {title}
          </h4>
          {subtitle && (
            <p className="text-sm text-gray-500 dark:text-gray-400 truncate mt-0.5">
              {subtitle}
            </p>
          )}
        </div>
      </Card>
    );
  }
);

MediaCard.displayName = 'MediaCard';

export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  MediaCard,
  cardVariants,
};
